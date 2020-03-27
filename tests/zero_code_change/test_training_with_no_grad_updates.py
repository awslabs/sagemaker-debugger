# Standard Library
import os
from tempfile import TemporaryDirectory

# Third Party
import torch
from tests.zero_code_change.utils import build_json
from torch.utils.data import DataLoader, IterableDataset

# First Party
from smdebug.trials import create_trial


class TestDataset(IterableDataset):
    """Test dataset to generate random samples."""

    def __init__(self, num_items, num_samples):
        self.num_items = num_items
        self.num_samples = num_samples

    def __iter__(self):
        def gen_rand(num_samples):
            num = 0
            while num < num_samples:
                yield torch.randint(self.num_items, size=(1,)).item()
                num += 1

        return gen_rand(self.num_samples)


class TestModel(torch.nn.Module):
    """Test model that tries to learn the mapping between index and embedding."""

    def __init__(self, embedding_size):
        super(TestModel, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(1, embedding_size, bias=True), torch.nn.Tanh()
        )
        self.fc.apply(init_weights("xavier"))

    def forward(self, x):
        return self.fc(x)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_weights(method):
    def _init_weights(m):
        if type(m) == torch.nn.Linear:
            if method == "xavier":
                torch.nn.init.xavier_uniform_(m.weight)
            elif method == "he":
                torch.nn.init.kaiming_uniform_(m.weight)
            else:
                raise Exception("Unsupported initialization method: {}".format(method))

    return _init_weights


def do_training():
    # Generate a matrix of fake random embeddings.
    N = 1000
    EMBEDDING_SIZE = 768
    embedding_matrix = torch.nn.Embedding.from_pretrained(
        torch.randn(N, EMBEDDING_SIZE, dtype=torch.float32)
    )
    embedding_matrix = embedding_matrix.to(get_device())

    # Setup dataset + data loader.
    BATCH_SIZE = 64
    NUM_BATCHES = 100
    train_dataset = TestDataset(num_items=N, num_samples=NUM_BATCHES * BATCH_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Setup model.
    model = TestModel(embedding_size=EMBEDDING_SIZE).to(get_device())

    # Setup criterion.
    criterion = torch.nn.CosineEmbeddingLoss()

    # Setup optimizer.
    optimizer = torch.optim.Adam(model.parameters())

    # Training loop.
    for batch in train_loader:
        # Lookup items in the embedding matrix.
        item_ids = batch
        lookup_embeddings = embedding_matrix(item_ids.to(embedding_matrix.weight.device))
        # Forward pass.
        x = batch.float().to(get_device()).view(-1, 1)
        output_embeddings = model(x)
        # Calculate loss.
        labels = torch.ones(
            lookup_embeddings.shape[0], dtype=torch.int32, device=embedding_matrix.weight.device
        )
        loss = criterion(output_embeddings, lookup_embeddings, labels)
        # Zero out gradients.
        optimizer.zero_grad()
        # Do backwards pass.
        loss.backward()
        # Update model.
        optimizer.step()


def test_training_with_no_grad_updates():
    temp_dir = TemporaryDirectory().name
    path = build_json(temp_dir, include_collections=["losses"], save_interval="1")
    os.environ["SMDEBUG_CONFIG_FILE_PATH"] = str(path)
    do_training()
    trial = create_trial(temp_dir)
    assert len(trial.steps()) == 99
