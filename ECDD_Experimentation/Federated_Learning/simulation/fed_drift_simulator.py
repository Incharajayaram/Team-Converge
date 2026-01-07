"""
Hierarchical Federated Drift Simulator.

Orchestrates full federated learning simulation with:
- Multiple student clients (Arduino Nicla)
- Multiple hub aggregators (Raspberry Pi)
- Central drift server
- Drift scenario injection
- Performance tracking
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Dict, List, Optional, Tuple
import time
from tqdm import tqdm
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from client.student_client import StudentClient
from client.client_monitor import ClientMonitor
from server.teacher_aggregator import TeacherHubAggregator
from server.drift_server import CentralDriftServer
from simulation.data_partitioning import (
    create_non_iid_splits, assign_students_to_hubs, resample_to_powerlaw
)
from simulation.drift_scenarios import DriftScenario


class HierarchicalFedSimulator:
    """
    Simulator for hierarchical federated learning with drift detection.
    
    Architecture:
    Students (Nicla) → Hubs (Pi) → Central Server
    """
    
    def __init__(self,
                 student_model: torch.nn.Module,
                 teacher_model: torch.nn.Module,
                 dataset: Dataset,
                 baseline_hist: np.ndarray,
                 num_hubs: int = 3,
                 students_per_hub: int = 5,
                 non_iid: bool = True,
                 powerlaw: bool = True,
                 dropout_rate: float = 0.2,
                 random_seed: int = 42):
        """
        Initialize hierarchical federated simulator.
        
        Args:
            student_model: Student model (Tiny LaDeDa)
            teacher_model: Teacher model (LaDeDa)
            dataset: Dataset to split among clients
            baseline_hist: Baseline distribution for drift detection
            num_hubs: Number of hub devices
            students_per_hub: Students per hub
            non_iid: Use non-IID data splits
            powerlaw: Use power-law data sizes
            dropout_rate: Client dropout rate per round
            random_seed: Random seed
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        self.num_hubs = num_hubs
        self.students_per_hub = students_per_hub
        self.num_students = num_hubs * students_per_hub
        self.dropout_rate = dropout_rate
        
        print(f"Initializing Hierarchical Federated Simulator...")
        print(f"  Hubs: {num_hubs}")
        print(f"  Students per hub: {students_per_hub}")
        print(f"  Total students: {self.num_students}")
        print(f"  Dataset size: {len(dataset)}")
        
        # Create data splits
        print(f"  Creating data splits (non-IID={non_iid}, power-law={powerlaw})...")
        self.dataset = dataset
        self.client_splits = self._create_client_splits(non_iid, powerlaw)
        
        # Assign students to hubs
        print(f"  Assigning students to hubs...")
        self.student_to_hub = assign_students_to_hubs(
            self.num_students, num_hubs, method='balanced', random_seed=random_seed
        )
        
        # Initialize central server
        print(f"  Initializing central server...")
        self.central_server = CentralDriftServer(
            baseline_hist=baseline_hist,
            num_hubs=num_hubs,
            min_hubs_for_aggregation=max(2, num_hubs // 2),
            target_fpr=0.01
        )
        
        # Initialize hubs
        print(f"  Initializing {num_hubs} hub aggregators...")
        self.hubs = self._create_hubs(teacher_model, baseline_hist)
        
        # Initialize students
        print(f"  Initializing {self.num_students} student clients...")
        self.students = self._create_students(student_model, baseline_hist)
        
        # Tracking
        self.current_round = 0
        self.round_logs = []
        self.drift_events = []
        
        print(f"✓ Simulator initialized")
    
    def _create_client_splits(self, non_iid: bool, powerlaw: bool) -> List[List[int]]:
        """Create data splits for clients."""
        if non_iid:
            splits = create_non_iid_splits(
                self.dataset,
                num_clients=self.num_students,
                concentration=0.5,
                random_seed=self.random_seed
            )
        else:
            # IID splits
            indices = np.arange(len(self.dataset))
            np.random.shuffle(indices)
            splits = np.array_split(indices, self.num_students)
            splits = [s.tolist() for s in splits]
        
        if powerlaw:
            splits = resample_to_powerlaw(
                splits,
                alpha=1.5,
                min_samples=50,
                random_seed=self.random_seed
            )
        
        return splits
    
    def _create_hubs(self, teacher_model: torch.nn.Module, baseline_hist: np.ndarray) -> List[TeacherHubAggregator]:
        """Create hub aggregators."""
        hubs = []
        for hub_id in range(self.num_hubs):
            hub = TeacherHubAggregator(
                teacher_model=teacher_model,
                central_server_url=f"mock://hub_{hub_id}",  # Mock URL for simulation
                hub_id=hub_id,
                min_students_for_aggregation=max(2, self.students_per_hub // 2),
                baseline_hist=baseline_hist
            )
            hubs.append(hub)
        
        return hubs
    
    def _create_students(self, student_model: torch.nn.Module, baseline_hist: np.ndarray) -> List[StudentClient]:
        """Create student clients."""
        students = []
        for student_id in range(self.num_students):
            # Find which hub this student belongs to
            hub_id = self._get_hub_for_student(student_id)
            
            student = StudentClient(
                student_model=student_model,
                hub_url=f"mock://hub_{hub_id}",  # Mock URL
                client_id=student_id,
                window_size=100,  # Smaller window for simulation
                epsilon=1.0,
                baseline_hist=baseline_hist
            )
            students.append(student)
        
        return students
    
    def _get_hub_for_student(self, student_id: int) -> int:
        """Get hub ID for a student."""
        for hub_id, student_ids in self.student_to_hub.items():
            if student_id in student_ids:
                return hub_id
        return 0  # Fallback
    
    def run_experiment(self,
                      num_rounds: int = 100,
                      scenario: Optional[DriftScenario] = None,
                      batch_size: int = 8,
                      predictions_per_round: int = 10,
                      verbose: bool = True) -> Dict:
        """
        Run full federated experiment.
        
        Args:
            num_rounds: Number of federated rounds
            scenario: Optional drift scenario to inject
            batch_size: Batch size for inference
            predictions_per_round: Predictions per client per round
            verbose: Print progress
            
        Returns:
            Dictionary with experiment results
        """
        print(f"\nStarting experiment:")
        print(f"  Rounds: {num_rounds}")
        print(f"  Scenario: {scenario.__class__.__name__ if scenario else 'None'}")
        print(f"  Predictions per round: {predictions_per_round}")
        
        for round_num in tqdm(range(num_rounds), desc="Federated Rounds"):
            self.current_round = round_num
            round_start_time = time.time()
            
            # Run round
            round_log = self.run_round(
                round_num=round_num,
                scenario=scenario,
                batch_size=batch_size,
                predictions_per_round=predictions_per_round
            )
            
            round_log['round_time'] = time.time() - round_start_time
            self.round_logs.append(round_log)
            
            # Print progress
            if verbose and round_num % 10 == 0:
                self._print_round_summary(round_num, round_log)
        
        # Compile results
        results = self._compile_results()
        
        print(f"\n✓ Experiment complete")
        print(f"  Total rounds: {num_rounds}")
        print(f"  Drift detected: {results['drift_detected']}")
        print(f"  Anomalous hubs: {len(results['anomalous_hubs'])}")
        
        return results
    
    def run_round(self,
                 round_num: int,
                 scenario: Optional[DriftScenario],
                 batch_size: int,
                 predictions_per_round: int) -> Dict:
        """
        Run a single federated round.
        
        Steps:
        1. Students make predictions on local data
        2. Students send sketches to hubs (when buffer full)
        3. Hubs aggregate and send to central server
        4. Central server detects drift
        5. Thresholds updated if drift detected
        
        Args:
            round_num: Current round number
            scenario: Drift scenario
            batch_size: Batch size
            predictions_per_round: Predictions per client
            
        Returns:
            Round log dictionary
        """
        # 1. Select active students (simulate dropout)
        active_students = self._sample_active_students()
        
        # 2. Students process local data
        student_predictions = 0
        sketches_sent = 0
        
        for student_id in active_students:
            student = self.students[student_id]
            
            # Get local data
            local_data = self._get_student_data(student_id, predictions_per_round)
            
            # Make predictions
            for image, label in local_data:
                # Apply drift scenario if applicable
                if scenario:
                    image = scenario.apply(image, round_num, student_id)
                
                # Convert PIL Image to tensor if needed
                if not isinstance(image, torch.Tensor):
                    image = self._pil_to_tensor(image)
                
                # Predict
                result = student.predict(image)
                student_predictions += 1
                
                # Check if sketch sent
                if student.monitor.is_buffer_full():
                    self._student_send_to_hub(student_id, student)
                    sketches_sent += 1
        
        # 3. Hubs aggregate if ready
        hubs_aggregated = 0
        for hub_id, hub in enumerate(self.hubs):
            if len(hub.student_sketches) >= hub.min_students:
                hub.aggregate_local_students()
                hubs_aggregated += 1
                
                # Send to central server (in simulation, call directly)
                self._hub_send_to_central(hub_id, hub)
        
        # 4. Central server checks drift
        drift_detected = self.central_server.drift_detected
        anomalous_hubs = self.central_server.anomalous_hubs.copy()
        
        # 5. Log drift events
        if drift_detected and round_num not in [e['round'] for e in self.drift_events]:
            self.drift_events.append({
                'round': round_num,
                'drift_scores': self.central_server.drift_history[-1] if self.central_server.drift_history else {},
                'anomalous_hubs': anomalous_hubs
            })
        
        # Compile round log
        round_log = {
            'round': round_num,
            'active_students': len(active_students),
            'student_predictions': student_predictions,
            'sketches_sent': sketches_sent,
            'hubs_aggregated': hubs_aggregated,
            'drift_detected': drift_detected,
            'anomalous_hubs': anomalous_hubs,
            'current_threshold': self.central_server.current_threshold
        }
        
        return round_log
    
    def _sample_active_students(self) -> List[int]:
        """Sample active students (simulate dropout)."""
        num_active = int(self.num_students * (1 - self.dropout_rate))
        return np.random.choice(self.num_students, size=num_active, replace=False).tolist()
    
    def _get_student_data(self, student_id: int, num_samples: int) -> List[Tuple]:
        """Get local data for a student."""
        indices = self.client_splits[student_id]
        
        # Sample indices
        if len(indices) < num_samples:
            # Oversample if needed
            sampled_indices = np.random.choice(indices, size=num_samples, replace=True)
        else:
            sampled_indices = np.random.choice(indices, size=num_samples, replace=False)
        
        # Get data
        data = []
        for idx in sampled_indices:
            item = self.dataset[int(idx)]
            if isinstance(item, (tuple, list)):
                data.append((item[0], item[1] if len(item) > 1 else 0))
            else:
                data.append((item, 0))
        
        return data
    
    def _pil_to_tensor(self, image) -> torch.Tensor:
        """Convert PIL Image to tensor."""
        from PIL import Image
        from torchvision import transforms
        
        if isinstance(image, Image.Image):
            transform = transforms.ToTensor()
            return transform(image)
        return image
    
    def _student_send_to_hub(self, student_id: int, student: StudentClient):
        """Student sends sketch to hub (simulated)."""
        hub_id = self._get_hub_for_student(student_id)
        hub = self.hubs[hub_id]
        
        # Generate sketch
        sketch = student.monitor.get_sketch(apply_dp=True, clear_after=True)
        
        # Hub receives sketch
        hub.receive_student_sketch(student_id, sketch)
    
    def _hub_send_to_central(self, hub_id: int, hub: TeacherHubAggregator):
        """Hub sends aggregation to central server (simulated)."""
        if hub.local_aggregated_hist is not None:
            self.central_server.receive_hub_aggregation(
                hub_id=hub_id,
                aggregated_hist=hub.local_aggregated_hist,
                metadata=hub.local_aggregated_stats
            )
    
    def _print_round_summary(self, round_num: int, round_log: Dict):
        """Print round summary."""
        print(f"\n  Round {round_num}:")
        print(f"    Active: {round_log['active_students']}/{self.num_students}")
        print(f"    Predictions: {round_log['student_predictions']}")
        print(f"    Drift: {round_log['drift_detected']}")
        if round_log['anomalous_hubs']:
            print(f"    Anomalous hubs: {round_log['anomalous_hubs']}")
    
    def _compile_results(self) -> Dict:
        """Compile experiment results."""
        return {
            'num_rounds': len(self.round_logs),
            'drift_detected': self.central_server.drift_detected,
            'drift_events': self.drift_events,
            'anomalous_hubs': self.central_server.anomalous_hubs,
            'round_logs': self.round_logs,
            'final_threshold': self.central_server.current_threshold,
            'total_predictions': sum(log['student_predictions'] for log in self.round_logs),
            'central_server_report': self.central_server.get_drift_report()
        }
    
    def save_results(self, filepath: str):
        """Save results to JSON file."""
        results = self._compile_results()
        
        # Convert numpy arrays to lists for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=convert)
        
        print(f"Results saved to {filepath}")


# Example usage
if __name__ == "__main__":
    print("Testing HierarchicalFedSimulator...")
    
    # Create mock models and dataset
    print("\n1. Creating mock components:")
    
    from torch import nn
    from torch.utils.data import TensorDataset
    
    class MockStudentModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(100, 1)
        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))
    
    class MockTeacherModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(100, 1)
        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))
    
    # Mock dataset
    images = torch.randn(1000, 3, 64, 64)
    labels = torch.randint(0, 2, (1000,))
    dataset = TensorDataset(images, labels)
    
    # Baseline distribution
    baseline_samples = np.random.beta(2, 5, size=5000)
    baseline_hist, _ = np.histogram(baseline_samples, bins=20, range=(0, 1))
    baseline_hist = baseline_hist / baseline_hist.sum()
    
    print("  ✓ Components created")
    
    # Create simulator
    print("\n2. Creating simulator:")
    simulator = HierarchicalFedSimulator(
        student_model=MockStudentModel(),
        teacher_model=MockTeacherModel(),
        dataset=dataset,
        baseline_hist=baseline_hist,
        num_hubs=2,
        students_per_hub=3,
        non_iid=True,
        powerlaw=True
    )
    
    # Run short experiment
    print("\n3. Running experiment (10 rounds):")
    results = simulator.run_experiment(
        num_rounds=10,
        scenario=None,
        predictions_per_round=5,
        verbose=False
    )
    
    print(f"\n4. Results:")
    print(f"  Total rounds: {results['num_rounds']}")
    print(f"  Total predictions: {results['total_predictions']}")
    print(f"  Drift detected: {results['drift_detected']}")
    
    print("\n✅ HierarchicalFedSimulator tests passed!")
