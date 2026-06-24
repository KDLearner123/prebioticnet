import torch
import torch.nn.functional as F
import random

def train_prebiotic_model(model, data, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_molecules = data['molecule'].num_nodes
    real_yields_edge = data.edge_index_dict[('reaction', 'yields', 'molecule')]
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # --- DATA AUGMENTATION ENVIRONMENT ---
        # Create a mutated copy of the graph for this epoch
        perturbed_data = data.clone()
        
        # 50% of the time, we simulate a hostile planet environment
        is_hostile_epoch = random.random() > 0.5
        
        if is_hostile_epoch:
            # Shift temperatures drastically up or down, and push pH to extreme bounds
            temp_shift = random.choice([-100.0, 100.0])
            ph_shift = random.choice([-3.5, 3.5])
            
            current_x = perturbed_data['reaction'].x.clone()
            current_x[:, 0] += temp_shift
            current_x[:, 1] += ph_shift
            perturbed_data['reaction'].x = current_x
            
            # Under hostile environments, real chemical pathways break down
            # We penalize the model if it scores them highly here (Targets drop to 0.0)
            targets_real = torch.zeros((real_yields_edge.shape[1], 1), device=real_yields_edge.device)
        else:
            # Standard optimal environment conditions
            targets_real = torch.ones((real_yields_edge.shape[1], 1), device=real_yields_edge.device)

        # 1. Forward pass on the configured epoch environment
        predictions_real = model(perturbed_data.x_dict, perturbed_data.edge_index_dict, yields_edge_index=real_yields_edge)
        
        # 2. Generate corrupted negative samples (structural noise)
        random_molecule_indices = torch.randint(0, num_molecules, (real_yields_edge.shape[1],), device=real_yields_edge.device)
        corrupted_yields_edge = real_yields_edge.clone()
        corrupted_yields_edge[1] = random_molecule_indices
        
        predictions_fake = model(perturbed_data.x_dict, perturbed_data.edge_index_dict, yields_edge_index=corrupted_yields_edge)
        targets_fake = torch.zeros_like(predictions_fake)
        
        # 3. Compute loss across structural and environmental dimensions
        loss_real = F.binary_cross_entropy_with_logits(predictions_real, targets_real)
        loss_fake = F.binary_cross_entropy_with_logits(predictions_fake, targets_fake)
        loss = loss_real + loss_fake
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            env_status = "Hostile" if is_hostile_epoch else "Optimal"
            print(f"Epoch {epoch+1:02d}/{epochs} | Env: {env_status:7s} | Loss: {loss.item():.4f}")
            
    return model