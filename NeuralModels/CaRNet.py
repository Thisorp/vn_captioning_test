# NeuralModels/CaRNet.py

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from typing import List
from .Dataset import MyDataset
from .Vocabulary import Vocabulary
from NeuralModels.Decoder.IDecoder import IDecoder
from NeuralModels.Encoder.IEncoder import IEncoder
from NeuralModels.Attention.IAttention import IAttention
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from VARIABLE import MAX_CAPTION_LENGTH
from .Metrics import Result
import pandas as pd  # Đảm bảo đã import pandas

class CaRNet(nn.Module):
    """
    The Convolutional and Recurrent Net (CaRNet).
    Supports various LSTM configurations and optional Attention mechanism.
    """

    def __init__(self, encoder: IEncoder, decoder: IDecoder, net_name: str, encoder_dim: int, hidden_dim: int, padding_index: int, vocab_size: int, embedding_dim: int, attention: IAttention = None, attention_dim: int = 1024, device: str = "cpu"):
        """Initialize the CaRNet model.

        Args:
            encoder (IEncoder): The encoder to use.
            decoder (IDecoder): The decoder to use.
            net_name (str): Name of the Neural Network.
            encoder_dim (int): Dimensionality of encoder output features.
            hidden_dim (int): Capacity of the LSTM hidden state.
            padding_index (int): Index of the padding token.
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of word embeddings.
            attention (IAttention, optional): Attention mechanism. Defaults to None.
            attention_dim (int, optional): Dimension of the attention layer. Defaults to 1024.
            device (str, optional): Device to run the model on. Defaults to "cpu".
        """
        super(CaRNet, self).__init__()
        self.padding_index = padding_index
        self.device = torch.device(device)
        self.name_net = net_name
        self.result_storer = Result()

        # Initialize Encoder and Decoder
        self.C = encoder(encoder_dim=encoder_dim, device=device)
        self.R = None

        # Initialize Attention if provided
        self.attention = False
        if attention is not None:
            self.attention = True
            self.R = decoder(hidden_dim, padding_index, vocab_size, embedding_dim, device, attention(self.C.encoder_dim, hidden_dim, attention_dim))
        else:
            self.R = decoder(hidden_dim, padding_index, vocab_size, embedding_dim, device)

        # Check if Recurrent network is initialized
        if self.R is None:
            raise ValueError("Could not create the Recurrent network.")

        # Move networks to the specified device
        self.C.to(self.device)
        self.R.to(self.device)

    def switch_mode(self, mode: str) -> bool:
        """Switch the model between training and evaluation modes.

        Args:
            mode (str): "training" or "evaluation".

        Returns:
            bool: True if mode is switched successfully, False otherwise.
        """
        if mode == "training":
            self.C.train()
            self.R.train()
            return True
        elif mode == "evaluation":
            self.C.eval()
            self.R.eval()
            return True
        return False

    def save(self, file_path: str) -> bool:
        """Save the model's state dictionaries.

        Args:
            file_path (str): Directory path to save the model.

        Returns:
            bool: True if saved successfully, False otherwise.
        """
        try:
            # Save Encoder
            torch.save(
                self.C.state_dict(),
                f"{file_path}/{self.name_net}_{self.C.encoder_dim}_{self.R.hidden_dim}_{self.R.attention.attention_dim if self.attention else 0}_C.pth"
            )
            # Save Decoder
            torch.save(
                self.R.state_dict(),
                f"{file_path}/{self.name_net}_{self.C.encoder_dim}_{self.R.hidden_dim}_{self.R.attention.attention_dim if self.attention else 0}_R.pth"
            )
        except Exception as ex:
            print(ex)
            return False
        return True

    def load(self, file_path: str) -> bool:
        """Load the model's state dictionaries from disk.

        Args:
            file_path (str): Directory path where the model is saved.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        try:
            # Load Encoder
            self.C.load_state_dict(
                torch.load(
                    f"{file_path}/{self.name_net}_{self.C.encoder_dim}_{self.R.hidden_dim}_{self.R.attention.attention_dim if self.attention else 0}_C.pth",
                    map_location=self.device
                )
            )
            # Load Decoder
            self.R.load_state_dict(
                torch.load(
                    f"{file_path}/{self.name_net}_{self.C.encoder_dim}_{self.R.hidden_dim}_{self.R.attention.attention_dim if self.attention else 0}_R.pth",
                    map_location=self.device
                )
            )
        except Exception as ex:
            print(ex)
            return False
        return True

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            images (torch.Tensor): Batch of images.
            captions (torch.Tensor): Batch of caption token IDs.

        Returns:
            torch.Tensor: Output scores for each token in the captions.
        """
        features = self.C(images)
        return self.R(features, captions)

    def __accuracy(self, outputs: torch.Tensor, labels: torch.Tensor, captions_length: List[int]) -> float:
        """Calculate accuracy using Jaccard Similarity.

        Args:
            outputs (torch.Tensor): Generated captions.
            labels (torch.Tensor): Ground truth captions.
            captions_length (list): Lengths of the captions.

        Returns:
            float: Accuracy score.
        """
        outputs = np.array(list(map(lambda output: np.unique(output), outputs.cpu())), dtype=object)
        labels = np.array(list(map(lambda label: np.unique(label), labels.cpu())), dtype=object)

        unions = [len(np.union1d(outputs[idx], labels[idx])) for idx in range(labels.shape[0])]
        intersections = [len(np.intersect1d(outputs[idx], labels[idx])) for idx in range(labels.shape[0])]

        return torch.mean(torch.tensor(intersections).float() / torch.tensor(unions).float())

    def train_model(self, train_set: MyDataset, validation_set: MyDataset, lr: float, epochs: int, vocabulary: Vocabulary):
        """Train the CaRNet model.

        Args:
            train_set (MyDataset): Training dataset.
            validation_set (MyDataset): Validation dataset.
            lr (float): Learning rate.
            epochs (int): Number of epochs.
            vocabulary (Vocabulary): Vocabulary object.
        """
        # Initialize Loss
        criterion = nn.CrossEntropyLoss(
            ignore_index=vocabulary.predefined_token_idx()["<START>"],
            reduction="sum"
        ).cuda() if self.device.type == "cuda" else nn.CrossEntropyLoss(
            ignore_index=vocabulary.predefined_token_idx()["<START>"],
            reduction="sum"
        )

        # Initialize optimizer
        optimizer = torch.optim.Adam(list(self.R.parameters()) + list(self.C.parameters()), lr)

        # Switch to training mode
        self.switch_mode("training")

        best_val_acc = -1.0
        best_epoch = -1

        for e in range(epochs):
            epoch_train_acc = 0.0
            epoch_train_loss = 0.0
            epoch_num_train_examples = 0
            batch_id_reporter = 0

            for images, captions_ids, captions_length in train_set:
                optimizer.zero_grad()

                batch_num_train_examples = images.shape[0]
                epoch_num_train_examples += batch_num_train_examples

                # Move data to device
                images = images.to(self.device)
                captions_ids = captions_ids.to(self.device)
                captions_length = captions_length.to(self.device)

                # Forward pass
                features = self.C(images)
                if not self.attention:
                    outputs, _ = self.R(features, captions_ids, captions_length)
                else:
                    outputs, _, alphas = self.R(features, captions_ids, captions_length)

                outputs = pack_padded_sequence(outputs, captions_length.cpu(), batch_first=True)
                targets = pack_padded_sequence(captions_ids, captions_length.cpu(), batch_first=True)
                loss = criterion(outputs.data, targets.data)

                # Add attention loss if attention is enabled
                if self.attention:
                    loss += float(torch.sum((
                        0.5 * torch.sum(
                            (1 - torch.sum(alphas, dim=1, keepdim=True)) ** 2,
                            dim=2,
                            keepdim=True
                        )
                    ), dim=0).squeeze(1))

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Evaluate training accuracy
                with torch.no_grad():
                    self.switch_mode("evaluation")
                    projections = self.C(images)

                    captions_output = torch.zeros((projections.shape[0], captions_ids.shape[1])).to(self.device)

                    for idx in range(projections.shape[0]):
                        if self.attention:
                            _caption_no_pad, _ = self.R.generate_caption(projections[idx].unsqueeze(0), captions_ids.shape[1])
                        else:
                            _caption_no_pad = self.R.generate_caption(projections[idx].unsqueeze(0), captions_ids.shape[1])
                        captions_output[idx, :_caption_no_pad.shape[1]] = _caption_no_pad

                    captions_output_padded = captions_output.type(torch.int32).to(self.device)
                    batch_train_acc = self.__accuracy(captions_output_padded.squeeze(1), captions_ids, captions_length)

                    epoch_train_acc += batch_train_acc * batch_num_train_examples
                    epoch_train_loss += loss.item() * batch_num_train_examples

                    self.switch_mode("training")

                    # Print mini-batch stats
                    print(f"  mini-batch:\tloss={loss.item():.4f}, tr_acc={batch_train_acc:.5f}")

                    # Store training info
                    self.result_storer.add_train_info(
                        epoch=int(e),
                        batch_id=int(batch_id_reporter),
                        loss=float(loss.item()),
                        accuracy=float(batch_train_acc)
                    )
                    batch_id_reporter += 1

            # Validate after each epoch
            val_acc = self.eval_net(validation_set, vocabulary)

            # Save model if validation accuracy improves
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = e + 1
                self.save("./.saved")

            epoch_train_loss /= epoch_num_train_examples
            epoch_train_acc /= epoch_num_train_examples

            # Store validation info
            self.result_storer.add_validation_info(epoch=int(e), accuracy=float(val_acc))

            # Print epoch stats
            print(f"epoch={e + 1}/{epochs}:\tloss={epoch_train_loss:.4f}, tr_acc={epoch_train_acc:.5f}, val_acc={val_acc:.5f}, {'BEST!' if best_epoch == e+1 else ''}")

            # Flush results to storage
            self.result_storer.flush()

    def eval_net(self, data_set, vocabulary):
        """Evaluate the model on a given dataset.

        Args:
            data_set (MyDataset): Dataset to evaluate.
            vocabulary (Vocabulary): Vocabulary object.

        Returns:
            float: Average accuracy over the dataset.
        """
        self.switch_mode("evaluation")
        acc = 0.0
        num_samples = 0

        with torch.no_grad():
            for images, captions_ids, captions_length in data_set:
                images = images.to(self.device)
                captions_ids = captions_ids.to(self.device)

                projections = self.C(images)
                captions_output = torch.zeros((projections.shape[0], captions_ids.shape[1])).to(self.device)

                for idx in range(projections.shape[0]):
                    if self.attention:
                        _caption_no_pad, _ = self.R.generate_caption(projections[idx].unsqueeze(0), captions_ids.shape[1])
                    else:
                        _caption_no_pad = self.R.generate_caption(projections[idx].unsqueeze(0), captions_ids.shape[1])
                    captions_output[idx, :_caption_no_pad.shape[1]] = _caption_no_pad

                captions_output_padded = captions_output.type(torch.int32).to(self.device)
                acc += self.__accuracy(captions_output_padded.squeeze(1), captions_ids, captions_length) * images.shape[0]
                num_samples += images.shape[0]

        self.switch_mode("training")
        return acc / num_samples

    def generate_caption(self, image: object, vocabulary: Vocabulary) -> str:
        """Generate a caption for a single image.

        Args:
            image (PIL.Image.Image or torch.Tensor): The image to caption.
            vocabulary (Vocabulary): Vocabulary object.

        Returns:
            str: Generated caption.

        Raises:
            ValueError: If the input is not a PIL Image or Tensor.
        """
        self.switch_mode("evaluation")

        # Preprocess the image if it's a PIL Image
        if isinstance(image, Image.Image):
            operations = transforms.Compose([
                transforms.Resize((MyDataset.image_transformation_parameter["crop"]["size"], MyDataset.image_transformation_parameter["crop"]["size"])),
                transforms.ToTensor(),
                transforms.Normalize(mean=MyDataset.image_transformation_parameter["mean"], std=MyDataset.image_transformation_parameter["std_dev"])
            ])
            image = operations(image)

        if not isinstance(image, torch.Tensor):
            raise ValueError(f"Image is not the expected type, got: {type(image)}.")

        # Generate caption
        features = self.C(image.unsqueeze(0))

        if self.attention:
            caption_ids, alphas = self.R.generate_caption(features, MAX_CAPTION_LENGTH)
            caption = vocabulary.rev_translate(caption_ids[0])
            # Optionally, generate attention visualization
            # self.__generate_image_attention(image, caption, alphas, image_name="attention.png")
        else:
            caption_ids = self.R.generate_caption(features, MAX_CAPTION_LENGTH)
            caption = vocabulary.rev_translate(caption_ids[0])

        self.switch_mode("training")
        return caption

    def __generate_image_caption(self, image: torch.Tensor, vocabulary: Vocabulary, image_name: str = "caption.png") -> str:
        """Generate and save an image with its caption.

        Args:
            image (torch.Tensor): Image tensor.
            vocabulary (Vocabulary): Vocabulary object.
            image_name (str, optional): Filename to save the image. Defaults to "caption.png".

        Returns:
            str: Generated caption.
        """
        self.switch_mode("evaluation")

        # Retrieve features
        features = self.C(image.unsqueeze(0))

        if self.attention:
            caption_ids, alphas = self.R.generate_caption(features, MAX_CAPTION_LENGTH)
            caption = vocabulary.rev_translate(caption_ids[0])
            # Optionally, generate attention visualization
            # self.__generate_image_attention(image, caption, alphas, image_name="attention.png")
        else:
            caption_ids = self.R.generate_caption(features, MAX_CAPTION_LENGTH)
            caption = vocabulary.rev_translate(caption_ids[0])

        # Adjust image colors (denormalize)
        image = image.clone()
        image[0] = image[0] * 0.229 + 0.485
        image[1] = image[1] * 0.224 + 0.456
        image[2] = image[2] * 0.225 + 0.406

        # Swap color channels
        image = image.permute(1, 2, 0)  # (H, W, C)

        # Save image with caption
        plt.figure(figsize=(15, 15))
        plt.imshow(image.cpu())
        plt.title(caption)
        plt.axis('off')
        plt.savefig(image_name)
        plt.close()

        self.switch_mode("training")
        return caption

    def __generate_image_attention(self, image: torch.Tensor, caption: List[str], alphas: torch.Tensor, image_name: str = "attention.png"):
        """Generate and save attention visualization on the image.

        Args:
            image (torch.Tensor): Image tensor.
            caption (List[str]): List of words in the caption.
            alphas (torch.Tensor): Attention weights.
            image_name (str, optional): Filename to save the attention image. Defaults to "attention.png".
        """
        self.switch_mode("evaluation")

        fig = plt.figure(figsize=(15, 15))
        _caption_len = len(caption)
        for t in range(_caption_len):
            # Reshape attention weights
            _att = alphas[t].reshape(self.R.attention.number_of_splits, self.R.attention.number_of_splits)

            # Add subplot for each word
            ax = fig.add_subplot((_caption_len + 1) // 2, 2, t + 1)
            ax.set_title(f"{caption[t]}", fontsize=12)

            img = ax.imshow(image.cpu())
            ax.imshow(_att, cmap='gray', alpha=0.7, extent=img.get_extent())

        plt.tight_layout()
        plt.savefig(image_name)
        plt.close()

        self.switch_mode("training")
