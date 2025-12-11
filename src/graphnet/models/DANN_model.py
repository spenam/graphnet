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
            #domain_classifier_layers.append(self._activation)
            #domain_classifier_layers.append(torch.nn.Dropout(self._dropout_readout))
            domain_classifier_layers.append(self.backbone._activation)
            domain_classifier_layers.append(torch.nn.Dropout(self.backbone._dropout_readout))

        self._domain_classifier = torch.nn.Sequential(*domain_classifier_layers)

    def compute_loss(
        self, preds: Tensor, data: List[Data], verbose: bool = False
    ) -> Tensor:
        """Compute and sum losses across tasks."""
        data_merged = {}
        target_labels_merged = list(set(self.target_labels))
        for label in target_labels_merged:
            #print(data)
            data_merged[label] = torch.cat([d[label] for d in [data]], dim=0)
            #data_merged[label] = torch.cat([dict([d])[label] for d in data if (len(d)>1) and (len(d)<5)], dim=0)
            #data_merged[label] = data[label]
        for task in self._tasks:
            if task._loss_weight is not None:
                #data_merged[task._loss_weight] = torch.cat(
                #    [d[task._loss_weight] for d in data], dim=0
                #)
                data_merged[task._loss_weight] = data[task._loss_weight]

        #preds = preds[0]
        #print("This is preds[0]")
        #print(preds[0])
        preds = torch.cat(preds,dim=0)
        #print("This is preds")
        #print(preds)
        #print("This is data_merged")
        #print(data_merged)


        losses = [
            task.compute_loss(preds, data_merged)
            for task in self._tasks
        ]
        if verbose:
            self.info(f"{losses}")
        assert all(
            loss.dim() == 0 for loss in losses
        ), "Please reduce loss for each task separately"
        return torch.sum(torch.stack(losses))

    def compute_loss_domain(
        self, preds: Tensor, data: List[Data], verbose: bool = False
    ) -> Tensor:
        """Compute and sum losses across tasks."""
        data_merged = {}
        target_labels_merged = ["is_data"] # a hard coded label, should be changed
        for label in target_labels_merged:
            data_merged[label] = torch.cat([d[label] for d in [data]], dim=0)
        for task in [self._domain_task]:
            if task._loss_weight is not None:
                #data_merged[task._loss_weight] = torch.cat(
                #    [d[task._loss_weight] for d in data], dim=0
                #)
                data_merged[task._loss_weight] = data[task._loss_weight]

        preds = torch.cat(preds,dim=0)


        losses = [
            task.compute_loss(preds, data_merged)
            for task in [self._domain_task]
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
            reverse_features = ReversalLayerF.apply(features, self._get_lambda_p())

            # main task predictions
            task_preds = self._main_task(features)
            task_preds_list.append(task_preds)

            # Domain predictions
            domain_preds = self._domain_classifier(reverse_features)
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
        mc_batch = batch["MC"]  
        real_data_batch = batch["RealData"]

        # Forward pass through the model
        preds_task_mc, preds_domain_mc = self(mc_batch)
        _, preds_domain_real_data = self(real_data_batch)
        print(" ")
        print("#######################") #This is just for checking dimensions
        print("##### info, will print 20 elements of each type #####")
        print("preds_task_mc")
        print(torch.flatten(preds_task_mc[0])[:20])
        print("preds_domain_mc")
        print(torch.flatten(preds_domain_mc[0])[:20])
        print("preds_domain_real_data")
        print(torch.flatten(preds_domain_real_data[0])[:20])

        # Split predictions for MC and RealData

        

        loss_task = self.compute_loss(preds_task_mc, mc_batch)
        loss_domain_mc = self.compute_loss_domain(preds_domain_mc, mc_batch)
        loss_domain_real_data = self.compute_loss_domain(preds_domain_real_data, real_data_batch)
        return loss_task, loss_domain_mc, loss_domain_real_data

    def training_step(
        self, train_batch: Union[Data, List[Data]], batch_idx: int
    ) -> Tensor:
        """Perform training step."""
        if isinstance(train_batch, Data):
            train_batch = [train_batch]
        loss_task, loss_domain_mc, loss_domain_real_data = self.shared_step(train_batch, batch_idx)

        mc_data = train_batch["MC"]  
        real_data = train_batch["RealData"]

        mc_data = mc_data.to_data_list()
        real_data = real_data.to_data_list()
        
        combined_data = mc_data + real_data
        combined_data = [Batch.from_data_list(combined_data)]

        
        train_batch = combined_data
        self.log_dict(
            {"train_loss_task": loss_task, "train_loss_domain_mc": loss_domain_mc, "train_loss_domain_real_data": loss_domain_real_data},
            batch_size=self._get_batch_size(train_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )

        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=True, on_step=True)
        return loss_task + loss_domain_mc + loss_domain_real_data

    def validation_step(
        self, val_batch: Union[Data, List[Data]], batch_idx: int
    ) -> Tensor:
        """Perform validation step."""
        if isinstance(val_batch, Data):
            val_batch = [val_batch]
        loss_task, loss_domain_mc, loss_domain_real_data = self.shared_step(val_batch, batch_idx)

        mc_data = val_batch["MC"]  
        real_data = val_batch["RealData"]

        mc_data = mc_data.to_data_list()
        real_data = real_data.to_data_list()
        
        combined_data = mc_data + real_data
        combined_data = [Batch.from_data_list(combined_data)]
        
        val_batch = combined_data

        self.log_dict(
            {"val_loss_task": loss_task, "val_loss_domain_mc": loss_domain_mc, "val_loss_domain_real_data": loss_domain_real_data},
            batch_size=self._get_batch_size(val_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return loss_task + loss_domain_mc + loss_domain_real_data

    def validate_tasks(self) -> None:
        """Verify that self._tasks contain compatible elements."""
        accepted_tasks = StandardLearnedTask
        for task in self._tasks:
            assert isinstance(task, accepted_tasks)

    def _get_p(self) -> float:
        p = self.global_step / self.trainer.estimated_stepping_batches
        return p

    def _get_lambda_p(self) -> float:
        p = self._get_p()
        lambda_p = 2.0 / (1.0 + np.exp(-1.0 * self._domain_gamma * p)) - 1

        return lambda_p
