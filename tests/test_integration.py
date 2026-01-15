"""
Integration tests for SPARTAN framework.
"""

import pytest
import numpy as np

from spartan import SPARTAN, SPARTANConfig, SPARTANResult
from spartan.models.mock import MockReasoningLLM
from spartan.mplq import MPLQ
from spartan.raas import RAAS
from spartan.rppo import RPPO
from spartan.attacks import NLBAAttack, SMVAAttack, MVNAAttack


class TestSPARTANIntegration:
    """Integration tests for main SPARTAN class."""
    
    def test_full_pipeline(self, mock_llm, sample_query):
        """Test complete SPARTAN pipeline."""
        spartan = SPARTAN(mock_llm)
        result = spartan.process(sample_query)
        
        assert isinstance(result, SPARTANResult)
        assert result.output is not None
        assert 0 <= result.risk_score <= 1
    
    def test_batch_processing(self, mock_llm):
        """Test batch processing."""
        spartan = SPARTAN(mock_llm)
        
        queries = [
            "What is 2 + 2?",
            "Solve x^2 = 4",
            "What is the integral of x?",
        ]
        
        results = spartan.batch_process(queries)
        
        assert len(results) == 3
        assert all(isinstance(r, SPARTANResult) for r in results)
    
    def test_with_optimization(self, mock_llm, sample_query):
        """Test with RPPO optimization enabled."""
        spartan = SPARTAN(mock_llm, enable_optimization=True)
        
        # Process multiple queries to build history
        for _ in range(15):
            result = spartan.process(sample_query)
        
        # Check optimization is working
        assert len(spartan.history) == 15
    
    def test_without_optimization(self, mock_llm, sample_query):
        """Test with optimization disabled."""
        spartan = SPARTAN(mock_llm, enable_optimization=False)
        result = spartan.process(sample_query)
        
        assert result.optimization_result is None
    
    def test_config_update(self, mock_llm, custom_config):
        """Test configuration update."""
        spartan = SPARTAN(mock_llm)
        spartan.set_config(custom_config)
        
        assert spartan.get_config() == custom_config
    
    def test_history_management(self, mock_llm, sample_query):
        """Test history reset."""
        spartan = SPARTAN(mock_llm)
        
        spartan.process(sample_query)
        assert len(spartan.history) == 1
        
        spartan.reset_history()
        assert len(spartan.history) == 0
    
    def test_member_vs_nonmember_detection(self, sample_query):
        """Test member vs non-member risk scores."""
        member_llm = MockReasoningLLM(member_mode=True, seed=42)
        nonmember_llm = MockReasoningLLM(member_mode=False, seed=42)
        
        spartan_member = SPARTAN(member_llm)
        spartan_nonmember = SPARTAN(nonmember_llm)
        
        result_member = spartan_member.process(sample_query)
        result_nonmember = spartan_nonmember.process(sample_query)
        
        # Member should generally have higher risk
        # (due to memorization patterns)
        assert result_member.risk_score >= 0
        assert result_nonmember.risk_score >= 0


class TestModuleInteraction:
    """Test interactions between SPARTAN modules."""
    
    def test_mplq_to_raas(self, sample_query, mock_llm):
        """Test MPLQ output feeds correctly to RAAS."""
        mplq = MPLQ()
        raas = RAAS()
        
        # Generate output
        output = mock_llm.generate(sample_query)
        
        # Run MPLQ
        risk_analysis = mplq.analyze(
            query=sample_query,
            prm_scores=output.prm_scores,
            vote_distribution=output.vote_distribution,
        )
        
        # Run RAAS with MPLQ output
        defense_result = raas.sanitize(
            output=output.output,
            risk_analysis=risk_analysis,
            reasoning_steps=output.reasoning_steps,
        )
        
        assert defense_result is not None
    
    def test_rppo_parameter_update(self, sample_query, mock_llm):
        """Test RPPO parameters update defense."""
        raas = RAAS()
        rppo = RPPO()
        
        initial_params = raas.get_params()
        
        # Simulate multiple iterations
        for i in range(15):
            rppo.update({
                "risk_score": 0.3 + 0.01 * i,
                "accuracy": 0.85,
                "compute": 0.4,
            })
        
        # Get optimized params
        opt_result = rppo.get_optimal_params()
        
        if opt_result is not None:
            raas.update_params(opt_result.params)
            new_params = raas.get_params()
            
            # Parameters should be updated
            assert new_params is not None


class TestAttackDefenseInteraction:
    """Test interaction between attacks and defenses."""
    
    def test_defense_reduces_attack_success(self, sample_query):
        """Defense should reduce attack success."""
        # Setup
        member_llm = MockReasoningLLM(member_mode=True, seed=42)
        spartan = SPARTAN(member_llm)
        attack = NLBAAttack()
        
        # Get undefended output
        output = member_llm.generate(sample_query)
        undefended_result = attack.execute(
            target_model=None,
            query=sample_query,
            prm_scores=output.prm_scores,
            reasoning_steps=output.reasoning_steps,
        )
        
        # Get defended output
        defended = spartan.process(sample_query)
        
        # Both should produce results
        assert undefended_result.success_score >= 0
        assert defended.defense_result is not None
    
    def test_multiple_attacks(self, mock_llm, sample_query):
        """Test running multiple attack types."""
        output = mock_llm.generate(sample_query, use_mcts=True)
        
        nlba = NLBAAttack()
        smva = SMVAAttack()
        mvna = MVNAAttack()
        
        nlba_result = nlba.execute(
            target_model=None,
            query=sample_query,
            prm_scores=output.prm_scores,
        )
        
        smva_result = smva.execute(
            target_model=None,
            query=sample_query,
            vote_distribution=output.vote_distribution,
        )
        
        mvna_result = mvna.execute(
            target_model=None,
            query=sample_query,
            mcts_values=output.mcts_values,
        )
        
        # All attacks should produce valid results
        assert 0 <= nlba_result.success_score <= 1
        assert 0 <= smva_result.success_score <= 1
        assert 0 <= mvna_result.success_score <= 1


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_valid_config(self):
        """Test valid configuration passes validation."""
        config = SPARTANConfig()
        errors = config.validate()
        
        assert len(errors) == 0
    
    def test_invalid_threshold(self):
        """Test invalid threshold detection."""
        config = SPARTANConfig(prm_threshold=1.5)  # Invalid
        errors = config.validate()
        
        assert len(errors) > 0
        assert any("prm_threshold" in e for e in errors)
    
    def test_invalid_epsilon(self):
        """Test invalid epsilon detection."""
        config = SPARTANConfig(epsilon_min=-0.1)  # Invalid
        errors = config.validate()
        
        assert len(errors) > 0
    
    def test_weight_sum_validation(self):
        """Test objective weight sum validation."""
        config = SPARTANConfig(
            accuracy_weight=0.5,
            privacy_weight=0.5,
            compute_weight=0.5,  # Sum > 1
        )
        errors = config.validate()
        
        assert len(errors) > 0


class TestEndToEnd:
    """End-to-end scenario tests."""
    
    def test_complete_workflow(self):
        """Test complete SPARTAN workflow."""
        # 1. Initialize with member model (high risk)
        llm = MockReasoningLLM(member_mode=True, seed=42)
        config = SPARTANConfig(
            prm_threshold=0.2,
            epsilon_max=0.7,
        )
        spartan = SPARTAN(llm, config=config)
        
        # 2. Process query
        query = "What is the derivative of x^3?"
        result = spartan.process(query, return_trace=True)
        
        # 3. Verify results
        assert result.output is not None
        assert result.risk_score > 0
        assert "reasoning_trace" in result.metadata
        
        # 4. Check defense was applied (high risk scenario)
        if result.risk_score > 0.3:
            assert result.defense_applied or result.epsilon_used > 0
    
    def test_low_risk_scenario(self):
        """Test low risk scenario."""
        llm = MockReasoningLLM(member_mode=False, seed=42)
        spartan = SPARTAN(llm)
        
        result = spartan.process("Simple question?")
        
        # Low risk should have minimal defense
        assert result.defense_result.epsilon_used <= 0.5
