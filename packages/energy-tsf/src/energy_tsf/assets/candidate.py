"""Candidate helpers backed by the explicit Memory Repository."""

from energy_tsf.contracts.artifact import CandidateRef, SolutionArtifact
from energy_tsf.memory.repository import MemoryRepository


def create_candidate(repository: MemoryRepository, solution: SolutionArtifact) -> CandidateRef:
    """Persist an immutable Candidate manifest for a saved Solution."""
    return repository.create_candidate(solution)


def read_candidate(repository: MemoryRepository, path: str) -> CandidateRef:
    """Load a Candidate and validate its manifest and Solution reference."""
    return repository.load_candidate(path)
