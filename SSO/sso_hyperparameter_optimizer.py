"""
SSO-based Hyperparameter Optimization for WSN-YOLO Training
Based on Shehadeh et al. Sperm Swarm Optimization algorithm
Optimizes YOLO training hyperparameters using current working values as baseline
"""

import numpy as np
import yaml
import os
import copy
import logging
from typing import Dict, List, Tuple, Any
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SSOHyperparameterOptimizer:
    """Sperm Swarm Optimization for YOLO training hyperparameter optimization
    
    Uses current working hyperparameters as baseline and optimizes around them
    to find the best combination for WSN-YOLO training
    """
    
    def __init__(self, n_sperms=20, max_iterations=25, dim=12):
        self.n_sperms = n_sperms
        self.max_iterations = max_iterations
        self.dim = dim  # Number of hyperparameters to optimize
        
        # SSO parameters (from original algorithm)
        self.velocity_coeff = 0.3
        self.log_min, self.log_max = 7, 14
        self.position_log_min, self.position_log_max = 35.5, 38.5
        
        # Current working hyperparameters (baseline from your fixed hyp.wsn.yaml)
        self.baseline_hyperparams = {
            'lr0': 0.01,
            'lrf': 0.1,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'box': 0.5,
            'cls': 0.5,
            'obj': 1.0,
            'iou_t': 0.25,
            'mosaic': 0.5,
            'mixup': 0.1,
            'fl_gamma': 0.0
        }
        
        # Define search bounds around baseline values
        self.bounds = self._define_hyperparameter_bounds()
        
        # Initialize sperm population around baseline
        self.population = self._initialize_population_around_baseline()
        
        # Initialize velocity
        self.velocity = self.velocity_coeff * np.random.randn(self.n_sperms, self.dim)
        
        self.fitness = np.full(self.n_sperms, float('inf'))
        self.personal_best = self.population.copy()
        self.personal_best_fitness = self.fitness.copy()
        self.global_best = None
        self.global_best_fitness = float('inf')
        self.convergence_curve = []
        
    def _define_hyperparameter_bounds(self):
        """Define search bounds around current working hyperparameters"""
        # Each parameter gets ±20-50% variation around baseline
        bounds = np.array([
            # Learning rate parameters
            [0.005, 0.02],      # lr0: 0.01 ± 50%
            [0.05, 0.2],        # lrf: 0.1 ± 50%
            [0.9, 0.95],        # momentum: 0.937 ± 2%
            [0.0001, 0.001],    # weight_decay: 0.0005 ± 50%
            [2.0, 5.0],         # warmup_epochs: 3.0 ± 33%
            
            # Loss function weights (critical - keep close to working values)
            [0.3, 0.7],         # box: 0.5 ± 20% (CRITICAL - don't go too low)
            [0.3, 0.7],         # cls: 0.5 ± 20%
            [0.7, 1.3],         # obj: 1.0 ± 30%
            [0.2, 0.35],        # iou_t: 0.25 ± 20%
            
            # Data augmentation (keep close to working values)
            [0.3, 0.7],         # mosaic: 0.5 ± 20%
            [0.05, 0.2],        # mixup: 0.1 ± 50%
            [0.0, 0.2],         # fl_gamma: 0.0 + variation
        ])
        return bounds
    
    def _initialize_population_around_baseline(self):
        """Initialize population with bias towards baseline values"""
        population = np.zeros((self.n_sperms, self.dim))
        
        # Convert baseline to parameter vector
        baseline_vector = self._hyperparams_to_vector(self.baseline_hyperparams)
        
        # Initialize population
        for i in range(self.n_sperms):
            if i == 0:
                # First sperm is exactly the baseline
                population[i] = baseline_vector
            elif i < self.n_sperms // 4:
                # 25% of population very close to baseline (±5%)
                noise = np.random.normal(0, 0.05, self.dim)
                population[i] = baseline_vector + noise * baseline_vector
            else:
                # Rest of population explores the full bounds
                population[i] = np.random.uniform(
                    self.bounds[:, 0], self.bounds[:, 1]
                )
        
        # Ensure all values are within bounds
        population = np.clip(population, self.bounds[:, 0], self.bounds[:, 1])
        
        return population
    
    def _hyperparams_to_vector(self, hyperparams):
        """Convert hyperparameter dict to optimization vector"""
        return np.array([
            hyperparams['lr0'],
            hyperparams['lrf'],
            hyperparams['momentum'],
            hyperparams['weight_decay'],
            hyperparams['warmup_epochs'],
            hyperparams['box'],
            hyperparams['cls'],
            hyperparams['obj'],
            hyperparams['iou_t'],
            hyperparams['mosaic'],
            hyperparams['mixup'],
            hyperparams['fl_gamma']
        ])
    
    def _vector_to_hyperparams(self, vector):
        """Convert optimization vector to hyperparameter dict"""
        base_hyperparams = copy.deepcopy(self.baseline_hyperparams)
        
        # Update optimized parameters
        base_hyperparams.update({
            'lr0': float(vector[0]),
            'lrf': float(vector[1]),
            'momentum': float(vector[2]),
            'weight_decay': float(vector[3]),
            'warmup_epochs': float(vector[4]),
            'box': float(vector[5]),
            'cls': float(vector[6]),
            'obj': float(vector[7]),
            'iou_t': float(vector[8]),
            'mosaic': float(vector[9]),
            'mixup': float(vector[10]),
            'fl_gamma': float(vector[11])
        })
        
        return base_hyperparams
    
    def optimize(self, fitness_function):
        """Main SSO optimization loop for hyperparameters"""
        logger.info(f"Starting SSO Hyperparameter Optimization with {self.n_sperms} sperms for {self.max_iterations} iterations")
        logger.info(f"Baseline hyperparameters: {self.baseline_hyperparams}")
        
        # Evaluate initial population
        for i in range(self.n_sperms):
            hyperparams = self._vector_to_hyperparams(self.population[i])
            self.fitness[i] = fitness_function(hyperparams)
            self.personal_best_fitness[i] = self.fitness[i]
            self.personal_best[i] = self.population[i].copy()
            
            if self.fitness[i] < self.global_best_fitness:
                self.global_best_fitness = self.fitness[i]
                self.global_best = self.population[i].copy()
        
        self.convergence_curve.append(self.global_best_fitness)
        logger.info(f"Initial best fitness: {self.global_best_fitness:.6f}")
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            logger.info(f"SSO Iteration {iteration + 1}/{self.max_iterations}, Best Fitness: {self.global_best_fitness:.6f}")
            
            # Update each sperm
            for i in range(self.n_sperms):
                # Boundary checking
                self.population[i] = np.clip(self.population[i], self.bounds[:, 0], self.bounds[:, 1])
                
                # Evaluate fitness
                hyperparams = self._vector_to_hyperparams(self.population[i])
                current_fitness = fitness_function(hyperparams)
                self.fitness[i] = current_fitness
                
                # Update personal best
                if current_fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = current_fitness
                    self.personal_best[i] = self.population[i].copy()
                
                # Update global best
                if current_fitness < self.global_best_fitness:
                    self.global_best_fitness = current_fitness
                    self.global_best = self.population[i].copy()
                    logger.info(f"New best fitness: {self.global_best_fitness:.6f}")
            
            # Update velocities and positions
            self._update_velocities_and_positions()
            
            self.convergence_curve.append(self.global_best_fitness)
            
            # Early stopping if no improvement
            if len(self.convergence_curve) > 8:
                recent_improvement = abs(self.convergence_curve[-8] - self.convergence_curve[-1])
                if recent_improvement < 1e-6:
                    logger.info(f"Early stopping at iteration {iteration + 1} due to convergence")
                    break
        
        return self.global_best, self.global_best_fitness
    
    def _update_velocities_and_positions(self):
        """Update velocities and positions using SSO equations"""
        for i in range(self.n_sperms):
            for j in range(self.dim):
                # SSO velocity update with logarithmic coefficients
                r1 = np.random.random()
                r2 = np.random.random()
                r3 = np.random.random()
                
                log_coeff1 = np.log10((self.log_max - self.log_min) * r1 + self.log_min)
                log_coeff2 = np.log10((self.log_max - self.log_min) * r2 + self.log_min)
                log_coeff3 = np.log10((self.position_log_max - self.position_log_min) * r3 + self.position_log_min)
                
                self.velocity[i, j] = (
                    r1 * log_coeff1 * self.velocity[i, j] +
                    log_coeff2 * log_coeff3 * (self.personal_best[i, j] - self.population[i, j]) +
                    log_coeff2 * log_coeff3 * (self.global_best[j] - self.population[i, j])
                )
        
        # Update positions
        self.population = self.population + self.velocity

class HyperparameterFitnessEvaluator:
    """Fitness evaluator for hyperparameter optimization
    
    Uses heuristic evaluation based on hyperparameter stability and training theory
    """
    
    def __init__(self):
        # Weights for different aspects of hyperparameter quality
        self.w_stability = 0.4      # Stability of training (no collapse)
        self.w_convergence = 0.3    # Convergence speed
        self.w_generalization = 0.2 # Generalization ability
        self.w_efficiency = 0.1     # Training efficiency
        
        # Known problematic ranges (from your training collapse analysis)
        self.critical_thresholds = {
            'box_min': 0.3,     # Below this causes localization collapse
            'mosaic_min': 0.2,  # Below this reduces generalization
            'lr0_max': 0.015,   # Above this causes instability
            'iou_t_min': 0.2,   # Below this causes training instability
        }
    
    def evaluate_fitness(self, hyperparams):
        """Evaluate hyperparameter configuration fitness"""
        try:
            # Extract key parameters
            lr0 = hyperparams['lr0']
            box = hyperparams['box']
            cls = hyperparams['cls']
            obj = hyperparams['obj']
            mosaic = hyperparams['mosaic']
            mixup = hyperparams['mixup']
            iou_t = hyperparams['iou_t']
            momentum = hyperparams['momentum']
            weight_decay = hyperparams['weight_decay']
            
            # 1. Stability Score (prevent training collapse)
            stability_score = self._calculate_stability_score(hyperparams)
            
            # 2. Convergence Score (learning efficiency)
            convergence_score = self._calculate_convergence_score(hyperparams)
            
            # 3. Generalization Score (avoid overfitting)
            generalization_score = self._calculate_generalization_score(hyperparams)
            
            # 4. Efficiency Score (training speed)
            efficiency_score = self._calculate_efficiency_score(hyperparams)
            
            # Combined fitness (lower is better)
            fitness = (
                self.w_stability * (1.0 - stability_score) +
                self.w_convergence * (1.0 - convergence_score) +
                self.w_generalization * (1.0 - generalization_score) +
                self.w_efficiency * (1.0 - efficiency_score)
            )
            
            # Heavy penalty for critical violations
            penalty = self._calculate_penalty(hyperparams)
            fitness += penalty
            
            logger.info(f"Hyperparameter Evaluation: Fitness={fitness:.6f}, "
                       f"Stability={stability_score:.3f}, Convergence={convergence_score:.3f}, "
                       f"Generalization={generalization_score:.3f}, Efficiency={efficiency_score:.3f}")
            
            return fitness
            
        except Exception as e:
            logger.error(f"Error evaluating hyperparameters: {e}")
            return float('inf')
    
    def _calculate_stability_score(self, hyperparams):
        """Calculate training stability score"""
        score = 1.0
        
        # Box loss weight (critical for localization)
        box = hyperparams['box']
        if box >= 0.4:
            score *= 1.0  # Optimal range
        elif box >= 0.3:
            score *= 0.8  # Acceptable
        else:
            score *= 0.3  # Risky (causes collapse)
        
        # IoU threshold
        iou_t = hyperparams['iou_t']
        if 0.2 <= iou_t <= 0.3:
            score *= 1.0  # Optimal
        else:
            score *= 0.7  # Suboptimal
        
        # Learning rate stability
        lr0 = hyperparams['lr0']
        if 0.008 <= lr0 <= 0.012:
            score *= 1.0  # Optimal
        elif lr0 > 0.015:
            score *= 0.6  # Too high (unstable)
        else:
            score *= 0.8  # Acceptable
        
        return min(1.0, score)
    
    def _calculate_convergence_score(self, hyperparams):
        """Calculate convergence efficiency score"""
        score = 0.5  # Base score
        
        # Learning rate for convergence
        lr0 = hyperparams['lr0']
        momentum = hyperparams['momentum']
        
        # Optimal learning rate range
        if 0.009 <= lr0 <= 0.011:
            score += 0.3
        elif 0.007 <= lr0 <= 0.013:
            score += 0.2
        else:
            score += 0.1
        
        # Momentum contribution
        if 0.93 <= momentum <= 0.94:
            score += 0.2
        else:
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_generalization_score(self, hyperparams):
        """Calculate generalization ability score"""
        score = 0.4  # Base score
        
        # Data augmentation contribution
        mosaic = hyperparams['mosaic']
        mixup = hyperparams['mixup']
        weight_decay = hyperparams['weight_decay']
        
        # Mosaic augmentation
        if 0.4 <= mosaic <= 0.6:
            score += 0.25  # Optimal
        elif 0.2 <= mosaic <= 0.8:
            score += 0.15  # Good
        else:
            score += 0.05  # Poor
        
        # Mixup augmentation
        if 0.05 <= mixup <= 0.15:
            score += 0.2   # Optimal
        elif mixup <= 0.25:
            score += 0.1   # Acceptable
        else:
            score += 0.05  # Too much
        
        # Weight decay regularization
        if 0.0003 <= weight_decay <= 0.0007:
            score += 0.15  # Optimal
        else:
            score += 0.05  # Suboptimal
        
        return min(1.0, score)
    
    def _calculate_efficiency_score(self, hyperparams):
        """Calculate training efficiency score"""
        score = 0.6  # Base score
        
        warmup_epochs = hyperparams['warmup_epochs']
        lrf = hyperparams['lrf']
        
        # Warmup efficiency
        if 2.5 <= warmup_epochs <= 3.5:
            score += 0.2  # Optimal
        else:
            score += 0.1  # Suboptimal
        
        # Learning rate schedule
        if 0.08 <= lrf <= 0.12:
            score += 0.2  # Optimal
        else:
            score += 0.1  # Suboptimal
        
        return min(1.0, score)
    
    def _calculate_penalty(self, hyperparams):
        """Calculate penalty for critical violations"""
        penalty = 0.0
        
        # Critical box loss threshold
        if hyperparams['box'] < self.critical_thresholds['box_min']:
            penalty += 5.0  # Heavy penalty
        
        # Critical mosaic threshold
        if hyperparams['mosaic'] < self.critical_thresholds['mosaic_min']:
            penalty += 2.0
        
        # Critical learning rate threshold
        if hyperparams['lr0'] > self.critical_thresholds['lr0_max']:
            penalty += 3.0
        
        # Critical IoU threshold
        if hyperparams['iou_t'] < self.critical_thresholds['iou_t_min']:
            penalty += 2.0
        
        return penalty

def save_optimized_hyperparameters(best_hyperparams, output_path="yolov7_maidul/data/hyp.wsn.sso_optimized.yaml"):
    """Save SSO-optimized hyperparameters to YAML file"""
    
    # Create complete hyperparameter file with all required parameters
    complete_hyperparams = {
        # Optimized parameters
        'lr0': best_hyperparams['lr0'],
        'lrf': best_hyperparams['lrf'],
        'momentum': best_hyperparams['momentum'],
        'weight_decay': best_hyperparams['weight_decay'],
        'warmup_epochs': best_hyperparams['warmup_epochs'],
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': best_hyperparams['box'],
        'cls': best_hyperparams['cls'],
        'cls_pw': 1.0,
        'obj': best_hyperparams['obj'],
        'obj_pw': 1.0,
        'iou_t': best_hyperparams['iou_t'],
        'anchor_t': 4.0,
        'fl_gamma': best_hyperparams['fl_gamma'],
        
        # Augmentation parameters
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.9,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': best_hyperparams['mosaic'],
        'mixup': best_hyperparams['mixup'],
        'copy_paste': 0.0,
        'paste_in': 0.0,
        'loss_ota': 1
    }
    
    # Add header comment
    header = """# SSO-Optimized Hyperparameters for WSN YOLO Training
# Generated using Sperm Swarm Optimization algorithm
# Optimized for ultra-lightweight models and WSN deployment
# Based on working baseline with SSO refinement

"""
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(header)
        yaml.dump(complete_hyperparams, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"SSO-optimized hyperparameters saved to: {output_path}")
    return output_path

def main():
    """Main SSO hyperparameter optimization process"""
    logger.info("Starting SSO-based Hyperparameter Optimization for WSN-YOLO")
    
    # Initialize optimizer and evaluator
    sso_optimizer = SSOHyperparameterOptimizer(n_sperms=15, max_iterations=20, dim=12)
    fitness_evaluator = HyperparameterFitnessEvaluator()
    
    # Run optimization
    best_params, best_fitness = sso_optimizer.optimize(fitness_evaluator.evaluate_fitness)
    
    # Convert best parameters to hyperparameter dict
    best_hyperparams = sso_optimizer._vector_to_hyperparams(best_params)
    
    # Save optimized hyperparameters
    output_path = save_optimized_hyperparameters(best_hyperparams)
    
    # Generate optimization report
    logger.info("=== SSO HYPERPARAMETER OPTIMIZATION COMPLETED ===")
    logger.info(f"Best Fitness: {best_fitness:.6f}")
    logger.info("Optimized Hyperparameters:")
    for key, value in best_hyperparams.items():
        logger.info(f"  {key}: {value:.6f}")
    logger.info(f"Saved to: {output_path}")
    
    # Compare with baseline
    logger.info("\n=== COMPARISON WITH BASELINE ===")
    baseline = sso_optimizer.baseline_hyperparams
    for key in baseline.keys():
        if key in best_hyperparams:
            change = ((best_hyperparams[key] - baseline[key]) / baseline[key]) * 100
            logger.info(f"{key}: {baseline[key]:.6f} → {best_hyperparams[key]:.6f} ({change:+.1f}%)")
    
    return best_hyperparams, best_fitness, output_path

if __name__ == "__main__":
    main()
