import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        # Determine device (CUDA, MPS for Apple Silicon, or CPU)
        if cuda:
            if t.cuda.is_available():
                self._device = t.device('cuda')
            elif t.backends.mps.is_available():
                self._device = t.device('mps')
            else:
                self._device = t.device('cpu')
                self._cuda = False

            self._model = model.to(self._device)
            self._crit = crit.to(self._device)
        else:
            self._device = t.device('cpu')

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss

        # Reset gradients
        self._optim.zero_grad()

        # Forward pass
        predictions = self._model(x)

        # Calculate loss
        loss = self._crit(predictions, y)

        # Backward pass
        loss.backward()

        # Update weights
        self._optim.step()

        return loss.item()

    def val_test_step(self, x, y):
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions

        # Forward pass
        predictions = self._model(x)

        # Calculate loss
        loss = self._crit(predictions, y)

        return loss.item(), predictions

    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it

        # Set training mode
        self._model.train()

        epoch_loss = 0.0
        num_batches = 0

        # Iterate through training set with progress bar
        for x, y in tqdm(self._train_dl, desc='Training', leave=False):
            # Transfer to GPU if available
            if self._cuda:
                x = x.to(self._device)
                y = y.to(self._device)

            # Perform training step
            loss = self.train_step(x, y)
            epoch_loss += loss
            num_batches += 1

        # Calculate average loss
        avg_loss = epoch_loss / num_batches
        return avg_loss

    def val_test(self):
        # set eval mode
        # disable gradient computation
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics
        # return the loss and print the calculated metrics

        # Set evaluation mode
        self._model.eval()

        epoch_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []

        # Disable gradient computation
        with t.no_grad():
            for x, y in tqdm(self._val_test_dl, desc='Validation', leave=False):
                # Transfer to GPU if available
                if self._cuda:
                    x = x.to(self._device)
                    y = y.to(self._device)

                # Perform validation step
                loss, predictions = self.val_test_step(x, y)
                epoch_loss += loss
                num_batches += 1

                # Save predictions and labels
                all_predictions.append(predictions.cpu())
                all_labels.append(y.cpu())

        # Calculate average loss
        avg_loss = epoch_loss / num_batches

        # Concatenate all predictions and labels
        all_predictions = t.cat(all_predictions, dim=0)
        all_labels = t.cat(all_labels, dim=0)

        # Convert predictions to binary (threshold at 0.5)
        binary_predictions = (all_predictions > 0.5).float()

        # Calculate F1 scores for each class
        f1_crack = f1_score(all_labels[:, 0].numpy(), binary_predictions[:, 0].numpy(), zero_division=0)
        f1_inactive = f1_score(all_labels[:, 1].numpy(), binary_predictions[:, 1].numpy(), zero_division=0)
        mean_f1 = (f1_crack + f1_inactive) / 2

        # Print metrics
        print(
            f'Validation Loss: {avg_loss:.4f}, F1 Crack: {f1_crack:.4f}, F1 Inactive: {f1_inactive:.4f}, Mean F1: {mean_f1:.4f}')

        return avg_loss

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch

        train_losses = []
        val_losses = []
        epoch_counter = 0
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        while True:
            # Stop by epoch number
            if epochs > 0 and epoch_counter >= epochs:
                break

            # Train for an epoch
            print(f'\nEpoch {epoch_counter + 1}')
            train_loss = self.train_epoch()
            train_losses.append(train_loss)

            # Calculate loss and metrics on validation set
            val_loss = self.val_test()
            val_losses.append(val_loss)

            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                self.save_checkpoint(epoch_counter)
                print(f'Checkpoint saved (best val loss: {best_val_loss:.4f})')
            else:
                epochs_without_improvement += 1

            epoch_counter += 1

            # Early stopping check
            if self._early_stopping_patience > 0:
                if epochs_without_improvement >= self._early_stopping_patience:
                    print(f'\nEarly stopping triggered after {epoch_counter} epochs')
                    break

        return train_losses, val_losses