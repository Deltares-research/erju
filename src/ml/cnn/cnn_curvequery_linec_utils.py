"""
Query-conditioned curve-prior CNN: model, dataset, loss functions.

Model: y_pred = c_hat - n_track*log(r/r0) + epsilon_hat(r)
  where epsilon_hat is learned as a function of event embedding and distance query.

Architecture:
  FO waveform -> shared encoder -> event embedding h_i
  metadata -> metadata embedding m_i
  [h_i, m_i] -> intensity head -> c_hat_i
  [h_i, m_i, query_features] -> residual head -> epsilon_hat_ij
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader


class CurvePriorCNN2D_Query(nn.Module):
    """
    Query-conditioned curve-prior CNN.
    
    Inputs:
      waveform: (B, 21, 7500) - 21 FO channels, 7500 time samples
      query_distance: (B,) - active track distance in meters
      track_id: (B,) - track number (0 or 1)
    
    Outputs:
      c_hat: (B,) - predicted event intensity
      epsilon_hat: (B,) - predicted residual at queried distance (or None if no residual head)
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        encoder_cfg = cfg.encoder
        head_cfg = cfg.head
        query_cfg = cfg.query_features
        
        # Shared 2D encoder
        self.encoder_layers = nn.ModuleList()
        
        in_channels = 1  # (B, 1, 21, 7500) with channel dim added later
        for i, out_channels in enumerate(encoder_cfg.conv_channels):
            layer = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=(encoder_cfg.kernel_ch[i], encoder_cfg.kernel_time[i]),
                    stride=(1, encoder_cfg.stride_time[i]),
                    padding="same"
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.encoder_layers.append(layer)
            in_channels = out_channels
        
        # Compute encoder output shape (for downstream heads)
        # After 4 conv layers with stride_time [2,2,2,2], time reduces: 7500 -> 937
        self.encoder_out_channels = encoder_cfg.conv_channels[-1]
        
        # Intensity head
        intensity_in = self.encoder_out_channels * 21 * 937  # flattened
        self.intensity_head = nn.Sequential()
        prev_dim = intensity_in
        
        for hidden_dim in head_cfg.intensity_hidden:
            self.intensity_head.append(nn.Linear(prev_dim, hidden_dim))
            self.intensity_head.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim
        
        self.intensity_head.append(nn.Linear(prev_dim, 1))  # single c_hat output
        
        # Residual head (optional)
        self.residual_head = None
        if cfg.train.enable_residual_head:
            # Embed query features
            query_n_features = query_cfg.n_query_features_minimal
            self.query_embedding = nn.Sequential(
                nn.Linear(query_n_features, head_cfg.query_embedding_dim),
                nn.ReLU(inplace=True)
            )
            
            # Concatenate event embedding + query embedding
            residual_in = intensity_in + head_cfg.query_embedding_dim
            
            self.residual_head = nn.Sequential()
            prev_dim = residual_in
            
            for hidden_dim in head_cfg.residual_hidden:
                self.residual_head.append(nn.Linear(prev_dim, hidden_dim))
                self.residual_head.append(nn.ReLU(inplace=True))
                prev_dim = hidden_dim
            
            self.residual_head.append(nn.Linear(prev_dim, 1))  # single epsilon output
    
    def forward(self, waveform, query_distance=None, track_id=None):
        """
        Args:
            waveform: (B, 21, 7500)
            query_distance: (B,) in meters (required for residual head)
            track_id: (B,) in {0, 1} (required for residual head)
        
        Returns:
            c_hat: (B,)
            epsilon_hat: (B,) or None
        """
        
        # Add channel dimension for 2D conv
        x = waveform.unsqueeze(1)  # (B, 1, 21, 7500)
        
        # Pass through encoder
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Flatten for dense layers
        event_embedding = x.reshape(x.shape[0], -1)  # (B, encoder_out_channels*21*time_steps)
        
        # Intensity head
        c_hat = self.intensity_head(event_embedding).squeeze(-1)  # (B,)
        
        # Residual head
        epsilon_hat = None
        if self.residual_head is not None and query_distance is not None:
            # Build query features
            n_batch = query_distance.shape[0]
            device = query_distance.device
            r0 = self.cfg.features.r0_ref
            
            log_r_ratio = torch.log(query_distance / r0)  # (B,)
            r_active_m = query_distance  # (B,)
            track_id_tensor = torch.tensor(track_id, dtype=torch.float32, device=device)  # (B,)
            
            query_features = torch.stack([
                log_r_ratio,
                r_active_m,
                track_id_tensor
            ], dim=1)  # (B, 3)
            
            # Embed query features
            query_embedding = self.query_embedding(query_features)  # (B, query_embedding_dim)
            
            # Concatenate with event embedding
            combined = torch.cat([event_embedding, query_embedding], dim=1)  # (B, event+query)
            
            epsilon_hat = self.residual_head(combined).squeeze(-1)  # (B,)
        
        return c_hat, epsilon_hat


class CurveDataset_Query(Dataset):
    """
    Event-level dataset for query-conditioned curve-prior model.
    
    Each sample represents one event and one sensor (query distance).
    """
    
    def __init__(self, waveforms, pgv_targets, distances, tracks, event_ids, 
                 sensor_names=None, split="train", holdout_sensor=None):
        """
        Args:
            waveforms: (n_events, 21, 7500)
            pgv_targets: (n_events, 5) per-sensor PGV values
            distances: (n_events, 5) per-sensor distances
            tracks: (n_events,) track IDs
            event_ids: (n_events,) event IDs
            sensor_names: list of 5 sensor names
            split: "train", "val", or "test"
            holdout_sensor: None or sensor name to exclude
        """
        
        self.waveforms = waveforms  # (n_events, 21, 7500)
        self.pgv_targets = pgv_targets  # (n_events, 5)
        self.distances = distances  # (n_events, 5)
        self.tracks = tracks  # (n_events,)
        self.event_ids = event_ids  # (n_events,)
        self.sensor_names = sensor_names or ["MP4", "MP8", "MP10", "MP1", "MP2"]
        self.split = split
        self.holdout_sensor = holdout_sensor
        
        # Build list of (event_idx, sensor_idx) tuples
        self.samples = []
        for event_idx in range(len(waveforms)):
            for sensor_idx, sensor_name in enumerate(self.sensor_names):
                # Skip held-out sensor if in training
                if self.holdout_sensor is not None and sensor_name == self.holdout_sensor:
                    if self.split == "train":
                        continue  # don't train on held-out sensor
                    # But include it for testing held-out generalization
                
                self.samples.append((event_idx, sensor_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        event_idx, sensor_idx = self.samples[idx]
        
        waveform = torch.from_numpy(self.waveforms[event_idx]).float()  # (21, 7500)
        pgv_target = float(self.pgv_targets[event_idx, sensor_idx])
        distance = float(self.distances[event_idx, sensor_idx])
        track = int(self.tracks[event_idx])
        event_id = int(self.event_ids[event_idx])
        sensor_name = self.sensor_names[sensor_idx]
        
        return {
            "waveform": waveform,
            "pgv_target": pgv_target,
            "distance": distance,
            "track": track,
            "event_id": event_id,
            "sensor_name": sensor_name,
            "sensor_idx": sensor_idx,
        }


def huber_loss_log(pred_log, target_log, delta=0.5, weights=None):
    """Huber loss in log space with optional sample weighting."""
    diff = pred_log - target_log
    abs_diff = torch.abs(diff)
    
    huber = torch.where(
        abs_diff <= delta,
        0.5 * diff ** 2,
        delta * (abs_diff - 0.5 * delta)
    )
    
    if weights is not None:
        huber = huber * weights
    
    return torch.mean(huber)


def compute_output_weights(batch_data, cfg):
    """
    Compute per-sample weights based on sensor (MP4) and optional PGV magnitude.
    
    Returns:
        weights: (B,) tensor
    """
    n_batch = len(batch_data["sensor_name"])
    weights = torch.ones(n_batch)
    
    # MP4 weighting
    mp4_weight = cfg.loss.mp4_weight
    if mp4_weight != 1.0:
        is_mp4 = np.array([s == "MP4" for s in batch_data["sensor_name"]])
        weights[is_mp4] = mp4_weight
    
    return weights


def compute_curve_target_intensity(pgv_log_targets, distances, n_track, r0=10.0):
    """
    Invert curve-prior to get event-level intensity target.
    
    From: log(PGV) = log(c) - n*log(r/r0)
    Solve for c: c_target = PGV * (r/r0)^n
    
    Or in log space: log(c) = log(PGV) + n*log(r/r0)
    
    Args:
        pgv_log_targets: (B,) log PGV values
        distances: (B,) distances in meters
        n_track: scalar attenuation exponent
        r0: reference distance (10 m)
    
    Returns:
        c_target_log: (B,) log-space intensity targets
    """
    
    log_r_ratio = np.log(distances / r0)
    c_target_log = pgv_log_targets + n_track * log_r_ratio
    
    return c_target_log
