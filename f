class CustomBCEWithLogitsLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        pos_weight: Optional[Tensor] = None,
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)
        self.weight: Optional[Tensor]
        self.pos_weight: Optional[Tensor]

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.pos_weight is not None:
            # Adjust the weight for positive samples
            pos_weight = self.pos_weight * target + (1 - target)
        else:
            pos_weight = torch.ones_like(target)

        # Use the adjusted pos_weight in the loss calculation
        return F.binary_cross_entropy_with_logits(
            input,
            target,
            self.weight,
            pos_weight=pos_weight,
            reduction=self.reduction,
        )
