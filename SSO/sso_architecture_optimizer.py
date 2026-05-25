"""
Sperm Swarm Optimization (SSO) based YOLO Architecture Optimizer for Wireless Sensor Networks
Based on Shehadeh et al. Sperm Swarm Optimization algorithm
Reference: "Sperm swarm optimization algorithm for optimizing wireless sensor network challenges"
Optimizes YOLO architecture for high accuracy with minimal computational power
"""

import numpy as np
import torch
import torch.nn as nn
import yaml
import time
import os
from pathlib import Path
import copy
from typing import Dict, List, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SSOOptimizer:
    """Sperm Swarm Optimization algorithm for YOLO architecture search
    
    Based on the original SSO paper by Shehadeh et al.:
    - Sperm population moves towards the egg (global optimum)
    - Uses biological-inspired movement patterns
    - Specifically designed for WSN optimization challenges
    """
    
    def __init__(self, n_sperms=30, max_iterations=50, dim=15):
        self.n_sperms = n_sperms  # Number of sperms in the population
        self.max_iterations = max_iterations
        self.dim = dim  # Number of optimization parameters
        
        # SSO parameters (from original Sperm Swarm Optimization paper)
        # These are the actual parameters from your SSO.m file
        self.velocity_coeff = 0.3  # Initial velocity coefficient
        self.log_min, self.log_max = 7, 14  # Logarithmic bounds for velocity update
        self.position_log_min, self.position_log_max = 35.5, 38.5  # Position update bounds
        
        # Problem bounds (architecture parameters)
        self.bounds = self._define_search_space()
        
        # Initialize sperm population
        self.population = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], 
            (self.n_sperms, self.dim)
        )
        
        # Initialize velocity (as in original SSO.m)
        self.velocity = self.velocity_coeff * np.random.randn(self.n_sperms, self.dim)
        
        self.fitness = np.full(self.n_sperms, float('inf'))
        self.personal_best = self.population.copy()
        self.personal_best_fitness = self.fitness.copy()
        self.global_best = None
        self.global_best_fitness = float('inf')
        self.convergence_curve = []
        
    def _define_search_space(self):
        """Define the search space for YOLO architecture optimization"""
        # Each row: [min_value, max_value] for each parameter
        bounds = np.array([
            # Backbone parameters
            [16, 64],     # Initial channels (0)
            [0.25, 1.0],  # Width multiplier P3 (1)
            [0.25, 1.0],  # Width multiplier P4 (2)
            [0.25, 1.0],  # Width multiplier P5 (3)
            
            # Head parameters
            [0.5, 2.0],   # Head depth multiplier (4)
            [16, 128],    # Head base channels (5)
            
            # Detection parameters
            [0.1, 0.9],   # Confidence threshold (6)
            [0.1, 0.9],   # NMS threshold (7)
            
            # Anchor parameters (normalized)
            [0.5, 2.0],   # Anchor scale P3 (8)
            [0.5, 2.0],   # Anchor scale P4 (9)
            [0.5, 2.0],   # Anchor scale P5 (10)
            
            # Optimization parameters
            [1, 4],       # Number of CSP blocks backbone (11)
            [1, 3],       # Number of CSP blocks head (12)
            [0, 1],       # Use separable conv (binary) (13)
            [0, 1],       # Use ghost modules (binary) (14)
        ])
        return bounds
    
    def optimize(self, fitness_function):
        """Main Sperm Swarm Optimization loop - following original SSO.m algorithm"""
        logger.info(f"Starting Sperm Swarm Optimization with {self.n_sperms} sperms for {self.max_iterations} iterations")
        
        # Evaluate initial population
        for i in range(self.n_sperms):
            self.fitness[i] = fitness_function(self.population[i])
            self.personal_best_fitness[i] = self.fitness[i]
            self.personal_best[i] = self.population[i].copy()
            
            if self.fitness[i] < self.global_best_fitness:
                self.global_best_fitness = self.fitness[i]
                self.global_best = self.population[i].copy()
        
        self.convergence_curve.append(self.global_best_fitness)
        
        # Main optimization loop (following original SSO.m structure)
        for iteration in range(self.max_iterations):
            logger.info(f"SSO Iteration {iteration + 1}/{self.max_iterations}, Best Fitness: {self.global_best_fitness:.6f}")
            
            # Update each sperm
            for i in range(self.n_sperms):
                # Boundary checking (as in original SSO.m)
                self.population[i] = np.clip(self.population[i], self.bounds[:, 0], self.bounds[:, 1])
                
                # Evaluate fitness
                current_fitness = fitness_function(self.population[i])
                self.fitness[i] = current_fitness
                
                # Update personal best
                if current_fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = current_fitness
                    self.personal_best[i] = self.population[i].copy()
                
                # Update global best
                if current_fitness < self.global_best_fitness:
                    self.global_best_fitness = current_fitness
                    self.global_best = self.population[i].copy()
                    
            # Update velocities and positions (as in original SSO.m)
            self._update_velocities_and_positions()
            
            self.convergence_curve.append(self.global_best_fitness)
            
            # Early stopping if no improvement
            if len(self.convergence_curve) > 10:
                recent_improvement = abs(self.convergence_curve[-10] - self.convergence_curve[-1])
                if recent_improvement < 1e-6:
                    logger.info(f"Early stopping at iteration {iteration + 1} due to convergence")
                    break
        
        return self.global_best, self.global_best_fitness
    
    def _update_velocities_and_positions(self):
        """Update velocities and positions using original SSO equations from SSO.m"""
        for i in range(self.n_sperms):
            for j in range(self.dim):
                # Original SSO velocity update equation from your SSO.m file
                r1 = np.random.random()
                r2 = np.random.random()
                r3 = np.random.random()
                
                # Calculate logarithmic coefficients as in original
                log_coeff1 = np.log10((self.log_max - self.log_min) * r1 + self.log_min)
                log_coeff2 = np.log10((self.log_max - self.log_min) * r2 + self.log_min)
                log_coeff3 = np.log10((self.position_log_max - self.position_log_min) * r3 + self.position_log_min)
                
                # SSO velocity update (from original SSO.m)
                self.velocity[i, j] = (
                    r1 * log_coeff1 * self.velocity[i, j] +
                    log_coeff2 * log_coeff3 * (self.personal_best[i, j] - self.population[i, j]) +
                    log_coeff2 * log_coeff3 * (self.global_best[j] - self.population[i, j])
                )
        
        # Update positions
        self.population = self.population + self.velocity

class YOLOArchitectureGenerator:
    """Generate YOLO architecture configurations from SSO parameters"""
    
    def __init__(self, base_config_path="yolov7/cfg/training/yolov7-tiny.yaml"):
        self.base_config_path = base_config_path
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
    
    def generate_architecture(self, sso_params):
        """Generate YOLO architecture from SSO parameters"""
        config = copy.deepcopy(self.base_config)
        
        # Extract parameters
        initial_channels = int(sso_params[0])
        width_mult_p3 = sso_params[1]
        width_mult_p4 = sso_params[2]
        width_mult_p5 = sso_params[3]
        head_depth_mult = sso_params[4]
        head_base_channels = int(sso_params[5])
        conf_thresh = sso_params[6]
        nms_thresh = sso_params[7]
        anchor_scale_p3 = sso_params[8]
        anchor_scale_p4 = sso_params[9]
        anchor_scale_p5 = sso_params[10]
        csp_blocks_backbone = int(sso_params[11])
        csp_blocks_head = int(sso_params[12])
        use_separable = sso_params[13] > 0.5
        use_ghost = sso_params[14] > 0.5
        
        # Update width multiplier
        config['width_multiple'] = min(width_mult_p3, width_mult_p4, width_mult_p5)
        
        # Update backbone with optimized channels
        backbone = self._optimize_backbone(
            config['backbone'], initial_channels, 
            width_mult_p3, width_mult_p4, width_mult_p5,
            csp_blocks_backbone, use_separable, use_ghost
        )
        config['backbone'] = backbone
        
        # Update head
        head = self._optimize_head(
            config['head'], head_base_channels, head_depth_mult,
            csp_blocks_head, use_separable, use_ghost
        )
        config['head'] = head
        
        # Update anchors
        config['anchors'] = self._optimize_anchors(
            config['anchors'], anchor_scale_p3, anchor_scale_p4, anchor_scale_p5
        )
        
        return config, {
            'conf_thresh': conf_thresh,
            'nms_thresh': nms_thresh,
            'flops_estimate': self._estimate_flops(config),
            'params_estimate': self._estimate_parameters(config)
        }
    
    def _optimize_backbone(self, backbone, initial_channels, w_p3, w_p4, w_p5, 
                          csp_blocks, use_separable, use_ghost):
        """Optimize backbone architecture"""
        optimized_backbone = []
        
        # Scale channel numbers based on width multipliers
        channel_scales = {
            'p1': 1.0,
            'p2': w_p3,
            'p3': w_p3,
            'p4': w_p4,
            'p5': w_p5
        }
        
        for i, layer in enumerate(backbone):
            new_layer = copy.deepcopy(layer)
            
            # Update channel numbers
            if len(layer) > 3 and isinstance(layer[3], list) and len(layer[3]) > 0:
                if isinstance(layer[3][0], int):
                    # Determine which stage we're in
                    if i < 8:  # P1-P2 stages
                        scale = channel_scales['p2']
                    elif i < 15:  # P3 stage
                        scale = channel_scales['p3']
                    elif i < 22:  # P4 stage
                        scale = channel_scales['p4']
                    else:  # P5 stage
                        scale = channel_scales['p5']
                    
                    new_layer[3][0] = max(8, int(layer[3][0] * scale))
            
            # Replace with separable convolutions if enabled
            if use_separable and layer[2] == 'Conv':
                if len(layer[3]) >= 2 and layer[3][1] == 3:  # 3x3 conv
                    new_layer[2] = 'DWConv'
            
            # Use Ghost modules if enabled
            if use_ghost and layer[2] == 'Conv':
                new_layer[2] = 'GhostConv'
            
            optimized_backbone.append(new_layer)
        
        return optimized_backbone
    
    def _optimize_head(self, head, base_channels, depth_mult, csp_blocks, 
                      use_separable, use_ghost):
        """Optimize head architecture"""
        optimized_head = []
        
        for layer in head:
            new_layer = copy.deepcopy(layer)
            
            # Update channel numbers
            if len(layer) > 3 and isinstance(layer[3], list) and len(layer[3]) > 0:
                if isinstance(layer[3][0], int):
                    new_layer[3][0] = max(8, int(layer[3][0] * depth_mult))
            
            # Replace with separable convolutions if enabled
            if use_separable and layer[2] == 'Conv':
                if len(layer[3]) >= 2 and layer[3][1] == 3:
                    new_layer[2] = 'DWConv'
            
            # Use Ghost modules if enabled
            if use_ghost and layer[2] == 'Conv':
                new_layer[2] = 'GhostConv'
            
            optimized_head.append(new_layer)
        
        return optimized_head
    
    def _optimize_anchors(self, anchors, scale_p3, scale_p4, scale_p5):
        """Optimize anchor configurations"""
        optimized_anchors = []
        scales = [scale_p3, scale_p4, scale_p5]
        
        for i, anchor_set in enumerate(anchors):
            scaled_anchors = []
            for j in range(0, len(anchor_set), 2):
                w = int(anchor_set[j] * scales[i])
                h = int(anchor_set[j+1] * scales[i])
                scaled_anchors.extend([w, h])
            optimized_anchors.append(scaled_anchors)
        
        return optimized_anchors
    
    def _estimate_flops(self, config):
        """Estimate FLOPs for the configuration"""
        # Simplified FLOP estimation based on architecture
        total_flops = 0
        
        # Backbone FLOPs estimation
        backbone_flops = 0
        for layer in config['backbone']:
            if len(layer) > 3 and isinstance(layer[3], list):
                if layer[2] == 'Conv' and len(layer[3]) >= 3:
                    channels, kernel, stride = layer[3][0], layer[3][1], layer[3][2]
                    # Simplified: channels * kernel^2 * output_size
                    output_size = 640 // (stride * 2)  # Rough estimation
                    backbone_flops += channels * (kernel**2) * (output_size**2)
        
        # Head FLOPs estimation
        head_flops = backbone_flops * 0.3  # Head is typically ~30% of backbone
        
        total_flops = backbone_flops + head_flops
        return total_flops / 1e9  # Convert to GFLOPs
    
    def _estimate_parameters(self, config):
        """Estimate number of parameters"""
        total_params = 0
        
        # Simplified parameter estimation
        for layer in config['backbone'] + config['head']:
            if len(layer) > 3 and isinstance(layer[3], list):
                if layer[2] in ['Conv', 'DWConv', 'GhostConv']:
                    if len(layer[3]) >= 2:
                        channels, kernel = layer[3][0], layer[3][1] if len(layer[3]) > 1 else 1
                        total_params += channels * (kernel**2) * 3  # Rough estimation
        
        return total_params / 1e6  # Convert to millions

class WSNFitnessEvaluator:
    """Fitness evaluator optimized for Wireless Sensor Networks"""
    
    def __init__(self, dataset_path="YOLO_Dataset", target_flops=1.0, target_params=2.0):
        self.dataset_path = dataset_path
        self.target_flops = target_flops  # Target GFLOPs
        self.target_params = target_params  # Target million parameters
        self.architecture_generator = YOLOArchitectureGenerator()
        
        # Weights for multi-objective optimization
        self.w_accuracy = 0.4
        self.w_flops = 0.3
        self.w_params = 0.2
        self.w_inference_time = 0.1
    
    def evaluate_fitness(self, sso_params):
        """Evaluate fitness of architecture defined by SSO parameters"""
        try:
            # Generate architecture
            config, metrics = self.architecture_generator.generate_architecture(sso_params)
            
            # Extract metrics
            estimated_flops = metrics['flops_estimate']
            estimated_params = metrics['params_estimate']
            
            # Penalty for exceeding constraints
            flops_penalty = max(0, (estimated_flops - self.target_flops) / self.target_flops)
            params_penalty = max(0, (estimated_params - self.target_params) / self.target_params)
            
            # Estimated accuracy (would require training for real accuracy)
            # For now, use heuristic based on architecture complexity
            estimated_accuracy = self._estimate_accuracy(config, estimated_flops, estimated_params)
            
            # Estimated inference time (based on FLOPs and architecture)
            estimated_inference_time = self._estimate_inference_time(estimated_flops, config)
            
            # Multi-objective fitness (lower is better)
            fitness = (
                self.w_accuracy * (1.0 - estimated_accuracy) +  # Maximize accuracy
                self.w_flops * (estimated_flops / self.target_flops) +  # Minimize FLOPs
                self.w_params * (estimated_params / self.target_params) +  # Minimize parameters
                self.w_inference_time * estimated_inference_time +  # Minimize inference time
                10 * (flops_penalty + params_penalty)  # Heavy penalty for constraint violation
            )
            
            # Log evaluation
            logger.info(f"Architecture Evaluation: Fitness={fitness:.6f}, "
                       f"Acc={estimated_accuracy:.3f}, FLOPs={estimated_flops:.3f}G, "
                       f"Params={estimated_params:.3f}M, InfTime={estimated_inference_time:.3f}ms")
            
            return fitness
            
        except Exception as e:
            logger.error(f"Error evaluating architecture: {e}")
            return float('inf')
    
    def _estimate_accuracy(self, config, flops, params):
        """Heuristic accuracy estimation based on architecture complexity"""
        # Base accuracy from architecture design
        base_accuracy = 0.7
        
        # Accuracy boost from model complexity (with diminishing returns)
        complexity_factor = min(flops * params, 2.0)  # Cap the complexity benefit
        accuracy_boost = 0.2 * (1 - np.exp(-complexity_factor))
        
        # Penalty for too few parameters (underfitting)
        if params < 0.5:
            accuracy_boost *= 0.5
        
        # Penalty for too many FLOPs without proportional parameters
        if flops > 2.0 and params < 1.0:
            accuracy_boost *= 0.7
        
        return min(0.95, base_accuracy + accuracy_boost)
    
    def _estimate_inference_time(self, flops, config):
        """Estimate inference time in milliseconds"""
        # Base time from FLOPs (assuming 1 GFLOP = 10ms on typical WSN hardware)
        base_time = flops * 10
        
        # Additional time from memory access patterns
        memory_overhead = len(config['backbone']) + len(config['head'])
        
        # Total estimated inference time
        total_time = base_time + memory_overhead * 0.5
        
        return total_time

def save_optimized_architecture(best_params, output_dir="optimized_architectures"):
    """Save the optimized architecture configuration"""
    os.makedirs(output_dir, exist_ok=True)
    
    generator = YOLOArchitectureGenerator()
    config, metrics = generator.generate_architecture(best_params)
    
    # Save configuration
    config_path = os.path.join(output_dir, "yolo_wsn_optimized.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "optimization_metrics.yaml")
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
    
    logger.info(f"Optimized architecture saved to {config_path}")
    logger.info(f"Metrics saved to {metrics_path}")
    
    return config_path, metrics_path

def main():
    """Main optimization process"""
    logger.info("Starting Sperm Swarm Optimization (SSO) based YOLO Architecture Optimization for WSN")
    
    # Initialize components  
    sso = SSOOptimizer(n_sperms=20, max_iterations=30, dim=15)
    evaluator = WSNFitnessEvaluator(target_flops=0.8, target_params=1.5)
    
    # Run optimization
    best_params, best_fitness = sso.optimize(evaluator.evaluate_fitness)
    
    # Save results
    config_path, metrics_path = save_optimized_architecture(best_params)
    
    # Generate final report
    generator = YOLOArchitectureGenerator()
    final_config, final_metrics = generator.generate_architecture(best_params)
    
    logger.info("=== OPTIMIZATION COMPLETED ===")
    logger.info(f"Best Fitness: {best_fitness:.6f}")
    logger.info(f"Estimated FLOPs: {final_metrics['flops_estimate']:.3f} G")
    logger.info(f"Estimated Parameters: {final_metrics['params_estimate']:.3f} M")
    logger.info(f"Configuration saved to: {config_path}")
    
    return best_params, best_fitness, final_config, final_metrics

if __name__ == "__main__":
    main() 