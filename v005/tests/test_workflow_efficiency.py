import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from v005.src.hybrid.model import HybridPrediction
from v005.src.hybrid.optimizer import nearest_measurement_distance, constraint_probability_target
from v005.src.hybrid.efficient_selection import select_efficient_candidates
from v005.src.search_config import SearchConfig, load_search_config, effective_search_config
from v005.src.settings import VariableDefinition, ProblemDefinition
from v005.src.stopping.metrics import predicted_optimum_index
from v005.src.workflow import run_trial


class WorkflowEfficiencyTests(unittest.TestCase):
    def test_nearest_distance_matches_reference_and_exact_zero(self):
        rng=np.random.default_rng(2);observed=rng.random((21,50));query=rng.random((300,50))
        query[:21]=observed
        expected=np.min(np.sqrt(np.sum((query[:,None]-observed[None])**2,axis=2)),axis=1)
        actual=nearest_measurement_distance(query,observed,memory_mb=1)
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(actual[:21],np.zeros(21))

    def test_single_remaining_slot_preserves_improvement(self):
        result=select_efficient_candidates(np.array([[0.],[1.]]),np.array([1.,.1]),
            np.array([1.,.02]),np.array([1.,1.]),1.,np.ones(2),np.ones(2),np.array([.5,.1]),
            1,SearchConfig(),has_feasible_observation=True,diversity_weight=.5)
        self.assertEqual(result.indices,[0]);self.assertEqual(result.roles,['IMPROVEMENT'])

    def test_fractional_ordinal_boundary_is_not_floored_down_by_roundoff(self):
        c=VariableDefinition('g','g','1','constraint','less_equal',None,None,None,.3)
        config=SearchConfig(ordinal_constraint_steps={'g':.1})
        self.assertAlmostEqual(constraint_probability_target(c,config),.35)

    def test_level_unit_requires_integer_raw_measurements_for_automatic_correction(self):
        constraint=VariableDefinition('g','g','level','constraint','less_equal',None,None,None,4.)
        problem=SimpleNamespace(constraints=[constraint])
        for observations,expected in [([3.,4.],4.5),([3.,3.5],4.)]:
            config=effective_search_config(SearchConfig(),problem,
                SimpleNamespace(result_rows={'g':observations}))
            self.assertEqual(constraint_probability_target(constraint,config),expected)

    def test_explicit_ordinal_step_overrides_inference(self):
        constraint=VariableDefinition('g','g','level','constraint','less_equal',None,None,None,4.)
        config=effective_search_config(SearchConfig(ordinal_constraint_steps={'g':.5}),
            SimpleNamespace(constraints=[constraint]),SimpleNamespace(result_rows={'g':[3.,3.5]}))
        self.assertEqual(constraint_probability_target(constraint,config),4.25)

    def test_predicted_optimum_uses_the_same_ordinal_boundary_as_recommendations(self):
        objective=VariableDefinition('y','y','mm','objective','minimize',None,None,None,None)
        constraint=VariableDefinition('g','g','level','constraint','less_equal',None,None,None,4.)
        problem=ProblemDefinition([],objective,[constraint],[],[objective,constraint],2)
        prediction=HybridPrediction(np.ones(2),{
            'y':{'hybrid_mean':np.array([1.,2.])},
            'g':{'hybrid_mean':np.array([4.4,3.9])}})
        self.assertEqual(predicted_optimum_index(prediction,problem,np.ones(2,dtype=bool)),1)
        self.assertEqual(predicted_optimum_index(prediction,problem,np.ones(2,dtype=bool),
            SearchConfig(ordinal_constraint_steps={'g':1.})),0)

    def test_settings_validation_does_not_accept_boolean_or_null_learning_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'settings.json'
            for value in [None,True,'30']:
                path.write_text(json.dumps({'max_learning_seconds':value}),encoding='utf-8')
                with self.assertRaises(ValueError):load_search_config(path)

    def exercise(self,mean_y,*,grid_exhausted=False):
        workspace=Path(__file__).resolve().parents[1]/'benchmarks'/'results'
        workspace.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='v005_efficiency_',dir=workspace) as tmp:
            trial=Path(tmp);(trial/'data').mkdir();(trial/'output').mkdir()
            self.assertEqual(trial.resolve().parent,workspace.resolve())
            (trial/'focus_settings.json').write_text(json.dumps({
                'initial_design_enabled': False, 'parameter_focus_enabled': False,
                'temporary_fix_enabled': False}), encoding='utf-8')
            (trial/'problem.csv').write_text(
                'column,display_name,unit,role,direction,lower,upper,step,target\n'
                'x,x,1,parameter,,0,1,0.1,\n'
                'y,y,1,objective,minimize,,,,\n'
                'g,g,1,constraint,less_equal,,,,0\n',encoding='utf-8')
            (trial/'data/experiments.csv').write_text('experiment_id,x,y,g\na,0,1,-1\nb,0.1,1.1,-1\n',encoding='utf-8')
            (trial/'search_settings.json').write_text(json.dumps({'candidate_mode':'budget'} if grid_exhausted else {}),encoding='utf-8')
            calls=[]
            def predict(x):
                calls.append(len(x));n=len(x)
                values={}
                for name,mean in [('y',mean_y),('g',-1.)]:
                    values[name]={'nn_pred':np.full(n,mean),'nn_std':np.full(n,1e-5),
                        'hybrid_mean':np.full(n,mean),'hybrid_std':np.full(n,1e-5),
                        'raw_hybrid_std':np.full(n,1e-5),'acquisition_std':np.full(n,1e-5),
                        'uncertainty_calibration_scale':np.ones(n)}
                return HybridPrediction(np.ones(n),values)
            model=SimpleNamespace(predict=predict,nn_model=SimpleNamespace(members=[None]),
                gp_models={'y':SimpleNamespace(y_scale=1.)},uncertainty_calibration_scale=1.,acquisition_uncertainty_scale=1.)
            training=SimpleNamespace(model=model,source_samples=2,retained_samples=2,learning_seconds=.001,
                                     nn_epochs=1,nn_normalized_mse=0.)
            with patch('v005.src.workflow.require_trial',return_value=trial), \
                 patch('v005.src.hybrid.model.train_hybrid_model',return_value=training) as fit, \
                 patch('v005.src.workflow.write_knowledge_diagnostics',return_value=[]), \
                 patch('v005.src.parameter_space.make_parameter_grid',return_value=np.array([[0.],[.1]]) if grid_exhausted else np.arange(11)[:,None]/10):
                run_trial('trial_test_efficiency',5)
            status=json.loads((trial/'output/stopping_status.json').read_text(encoding='utf-8'))
            budget=json.loads((trial/'output/search_budget.json').read_text(encoding='utf-8'))
            recs=[]
            if (trial/'output/recommendations.csv').exists():
                with (trial/'output/recommendations.csv').open(encoding='utf-8-sig',newline='') as f:recs=list(csv.DictReader(f))
            return status,budget,recs,calls,fit.call_count

    def test_empty_useful_batch_requires_review_without_false_convergence(self):
        status,budget,recs,calls,_=self.exercise(1.2)
        self.assertEqual(status['status'],'REVIEW_REQUIRED');self.assertFalse(status['metrics']['converged'])
        self.assertEqual(recs,[]);self.assertEqual(calls,[11]);self.assertEqual(budget['prediction_grid_passes'],1)

    def test_full_prediction_is_shared_and_recommendations_are_published_when_continuing(self):
        status,budget,recs,calls,_=self.exercise(.5)
        self.assertEqual(status['status'],'CONTINUE');self.assertEqual(calls,[11])
        self.assertGreater(len(recs),0);self.assertLessEqual(len(recs),5)
        self.assertEqual(int(recs[0]['actual_recommendations']),len(recs))
        self.assertEqual(budget['runtime']['requested_cpu_threads'],1)

    def test_sampled_candidate_exhaustion_skips_training_and_does_not_claim_full_search(self):
        status,budget,recs,calls,fit_count=self.exercise(.5,grid_exhausted=True)
        self.assertEqual(status['status'],'REVIEW_REQUIRED');self.assertIsNone(status['metrics']['all_candidates_explored'])
        self.assertEqual(fit_count,0);self.assertEqual(calls,[]);self.assertEqual(recs,[])
        self.assertTrue(budget['learning_skipped'])


if __name__=='__main__':unittest.main()
