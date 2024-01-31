def consistency_loss(vector_1, vector_2):
    return torch.sqrt(nn.MSELoss()(vector_1, vector_2))