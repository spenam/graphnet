"""Standard model class(es)."""

from typing import Any, Dict, List, Optional, Union, Type, Tuple
import torch
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch.optim import Adam
from torch.autograd import Function
import numpy as np

from graphnet.models.gnn.gnn import GNN
from graphnet.models import Model
from .easy_model import EasySyntax
from graphnet.models.task import StandardLearnedTask
from graphnet.models.graphs import GraphDefinition


class ReversalLayerF(
    Function
):  # As in https://github.com/fungtion/DANN_py3/blob/master/functions.py

    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        output = grad_output.neg() * ctx.alpha

        return output, None


class DANN_model(EasySyntax):
    """A Standard way of combining model components in GraphNeT.

    This model is compatible with the vast majority of supervised learning
    tasks such as regression, binary and multi-label classification.

    Capable of producing both event-level and pulse-level predictions.
    """

    def __init__(
        self,
        graph_definition: GraphDefinition,
        tasks: Union[StandardLearnedTask, List[StandardLearnedTask]],
        domain_task: StandardLearnedTask,
        backbone: Model = None,
        gnn: Optional[GNN] = None,
        optimizer_class: Type[torch.optim.Optimizer] = Adam,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
        backbone_output: Optional[int] = 256,
        domain_gamma: Optional[float] = 10,
        dummy_value: Optional[float] = 99999998,
        domain_classifier_layer_sizes: Optional[List[int]] = [256],
        main_task_layer_sizes: Optional[List[int]] = [256],
        max_epochs: Optional[int] = 100,
    ) -> None:
        """Construct `StandardModel`."""
        # Base class constructor
        super().__init__(
            tasks=tasks,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            scheduler_config=scheduler_config,
        )

        # deprecation warnings
        if (backbone is None) & (gnn is not None):
            backbone = gnn
            # Code continues after warning
            self.warning(
                "DeprecationWarning: Argument `gnn` will be deprecated in"
                " GraphNeT 2.0. Please use `backbone` instead."
                ""
            )
        elif (backbone is None) & (gnn is None):
            # Code stops
            raise TypeError("__init__() missing 1 required keyword argument:'backbone'")

        # Checks
        assert isinstance(backbone, Model)
        assert isinstance(graph_definition, GraphDefinition)

        # Member variable(s)
        self._graph_definition = graph_definition
        self.backbone = backbone
        self._backbone_output = backbone_output
        self._domain_gamma = domain_gamma
        self._dummy_value = dummy_value
        self._domain_task = domain_task
        self._domain_classifier_layer_sizes = domain_classifier_layer_sizes
        self._main_task_layer_sizes = main_task_layer_sizes
        self._max_epochs = max_epochs


        self._build_main_task()
        self._build_domain_classifier()

    def _build_main_task(self) -> None:
        """Build the main task of the network."""
        #nb_poolings = (
        #    len(self._global_pooling_schemes) if self._global_pooling_schemes else 1
        #)
        nb_poolings = 1
        nb_latent_features = self._backbone_output * nb_poolings

        main_task_layers = []
        layer_sizes = [nb_latent_features] + list(self._main_task_layer_sizes)
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            main_task_layers.append(torch.nn.Linear(nb_in, nb_out))
            #main_task_layers.append(self._activation)
            #main_task_layers.append(torch.nn.Dropout(self._dropout_readout))
            main_task_layers.append(self.backbone._activation)
            main_task_layers.append(torch.nn.Dropout(self.backbone._dropout_readout))

        self._main_task = torch.nn.Sequential(*main_task_layers)

    def _build_domain_classifier(self) -> None:
        """Build the domain classifier network."""

        #nb_poolings = (
        #    len(self._global_pooling_schemes) if self._global_pooling_schemes else 1
        #)
        nb_poolings = 1
        nb_latent_features = self._backbone_output * nb_poolings

        domain_classifier_layers = []
        layer_sizes = [nb_latent_features] + list(self._domain_classifier_layer_sizes)
        for nb_in, nb_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            domain_classifier_layers.append(torch.nn.Linear(nb_in, nb_out))
            domain_classifier_layers.append(self._activation)
            domain_classifier_layers.append(torch.nn.Dropout(self._dropout_readout))

        self._domain_classifier = torch.nn.Sequential(*domain_classifier_layers)

    def compute_loss(
        self, preds: Tensor, data: List[Data], verbose: bool = False
    ) -> Tensor:
        """Compute and sum losses across tasks."""
        data_merged = {}
        target_labels_merged = list(set(self.target_labels))
        for label in target_labels_merged:
            data_merged[label] = torch.cat([d[label] for d in data], dim=0)
        for task in self._tasks:
            if task._loss_weight is not None:
                data_merged[task._loss_weight] = torch.cat(
                    [d[task._loss_weight] for d in data], dim=0
                )

        losses = [
            task.compute_loss(pred, data_merged)
            for task, pred in zip(self._tasks, preds)
        ]
        if verbose:
            self.info(f"{losses}")
        assert all(
            loss.dim() == 0 for loss in losses
        ), "Please reduce loss for each task separately"
        return torch.sum(torch.stack(losses))

    def forward(
        self, data: Union[Data, List[Data]]
    ) -> List[Union[Tensor, Data, Tuple[Tensor, Tensor]]]:
        """Forward pass, chaining model components."""
        if isinstance(data, Data):
            data = [data]

        task_preds_list = []
        domain_preds_list = []

        for d in data:

            features = self.backbone(d)

            # main task predictions
            task_preds = self._main_task(features)
            task_preds_list.append(task_preds)

            # Domain predictions
            domain_preds = self._domain_classifier(features)
            domain_preds = ReversalLayerF.apply(domain_preds, self._get_lambda_p())
            domain_preds_list.append(domain_preds)

        x_task = torch.cat(task_preds_list, dim=0)
        x_domain = torch.cat(domain_preds_list, dim=0)

        preds_task = [task(x_task) for task in self._tasks]
        preds_domain = [task(x_domain) for task in [self._domain_task]]
        return preds_task, preds_domain

    def shared_step(self, batch: List[Data], batch_idx: int) -> Tensor:
        """Perform shared step.

        Applies the forward pass and the following loss calculation, shared
        between the training and validation step.
        """
        x_s, _ = batch["MC"]
        x_t, _ = batch["RealData"]
        x = torch.cat([x_s, x_t], dim=0)
        preds = self(x)

        y_s, _ = preds[0].chunk(2, dim=0)

        loss_task = self.compute_loss(y_s, batch["MC"])
        loss_domain = self.compute_loss(preds[1], batch)
        return loss_task + loss_domain

    def validate_tasks(self) -> None:
        """Verify that self._tasks contain compatible elements."""
        accepted_tasks = StandardLearnedTask
        for task in self._tasks:
            assert isinstance(task, accepted_tasks)

    def _get_p(self) -> float:
        current_iterations = self.global_step
        current_epoch = self.current_epoch
        #len_dataloader = len(self.train_dataloader())
        p = (
            float(current_iterations + current_epoch)# * len_dataloader)
            / self._max_epochs
            #/ len_dataloader
        )

        return p

    def _get_lambda_p(self) -> float:
        p = self._get_p()
        lambda_p = 2.0 / (1.0 + np.exp(-1.0 * self._domain_gamma * p)) - 1

        return lambda_p
