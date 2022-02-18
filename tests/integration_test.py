"""Perform integration tests for `orion.algo.nevergrad`."""
import nevergrad as ng
import pytest
from orion.benchmark.task.branin import Branin
from orion.core.utils import backward
from orion.testing.algo import BaseAlgoTests, phase

WORKING = [
    "NGO",
    "NGOpt",
    "NGOpt10",
    "NGOpt12",
    "NGOpt13",
    "NGOpt14",
    "NGOpt15",
    "NGOpt16",
    "NGOpt21",
    "NGOpt36",
    "NGOpt38",
    "NGOpt39",
    "NGOpt4",
    "NGOpt8",
    "NGOptBase",
    "TBPSA",
    "TripleCMA",
    "TwoPointsDE",
    "cGA",
    "CMA",
    "DE",
    "DiagonalCMA",
    "MetaModel",
    "MixES",
    "MultiCMA",
    "MutDE",
    "NaiveTBPSA",
    "NoisyDE",
    "ORandomSearch",
    "PolyCMA",
    "QORandomSearch",
    "RandomSearch",
    "RealSpacePSO",
    "RecES",
    "RecMixES",
    "RecMutDE",
    "RescaledCMA",
    "RotatedTwoPointsDE",
    "Shiwa",
]

NOT_WORKING = [
    "ASCMADEthird",
    "AdaptiveDiscreteOnePlusOne",
    "AlmostRotationInvariantDE",
    "AnisotropicAdaptiveDiscreteOnePlusOne",
    "AvgMetaRecenteringNoHull",
    "BO",
    "BOSplit",
    "BayesOptimBO",
    "CM",
    "CMandAS2",
    "CMandAS3",
    "CauchyLHSSearch",
    "CauchyOnePlusOne",
    "CauchyScrHammersleySearch",
    "CmaFmin2",
    "DiscreteBSOOnePlusOne",
    "DiscreteDoerrOnePlusOne",
    "DiscreteLenglerOnePlusOne",
    "DiscreteOnePlusOne",
    "DoubleFastGADiscreteOnePlusOne",
    "EDA",
    "ES",
    "FCMA",
    "GeneticDE",
    "HaltonSearch",
    "HaltonSearchPlusMiddlePoint",
    "HammersleySearch",
    "HammersleySearchPlusMiddlePoint",
    "HullAvgMetaRecentering",
    "HullAvgMetaTuneRecentering",
    "LHSSearch",
    "LargeHaltonSearch",
    "LhsDE",
    "MetaModelOnePlusOne",
    "MetaRecentering",
    "MetaTuneRecentering",
    "MultiDiscrete",
    "MultiScaleCMA",
    "NaiveIsoEMNA",
    "NelderMead",
    "NoisyBandit",
    "NoisyDiscreteOnePlusOne",
    "NoisyOnePlusOne",
    "NonNSGAIIES",
    "OScrHammersleySearch",
    "OnePlusOne",
    "OptimisticDiscreteOnePlusOne",
    "OptimisticNoisyOnePlusOne",
    "PCABO",
    "PSO",
    "ParaPortfolio",
    "Portfolio",
    "PortfolioDiscreteOnePlusOne",
    "Powell",
    "PymooNSGA2",
    "QOScrHammersleySearch",
    "QrDE",
    "RPowell",
    "RSQP",
    "RandomSearchPlusMiddlePoint",
    "RecombiningPortfolioDiscreteOnePlusOne",
    "RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne",
    "RotationInvariantDE",
    "SPSA",
    "SQP",
    "SQPCMA",
    "ScrHaltonSearch",
    "ScrHaltonSearchPlusMiddlePoint",
    "ScrHammersleySearch",
    "ScrHammersleySearchPlusMiddlePoint",
    "SparseDoubleFastGADiscreteOnePlusOne",   
]

HANGING = [
    "ChainCMAPowell",
    "ChainDiagonalCMAPowell",
    "ChainMetaModelPowell",
    "ChainMetaModelSQP",
    "ChainNaiveTBPSACMAPowell",
    "ChainNaiveTBPSAPowell",
    "Cobyla",
    "RCobyla",
]

MODEL_NAMES = WORKING

@pytest.fixture(autouse=True, params=MODEL_NAMES)
def _config(request):
    """ Fixture that parametrizes the configuration used in the tests below. """
    if ng.optimizers.registry[request.param].no_parallelization:
        num_workers = 1
    else:
        num_workers = 10
    TestNevergradOptimizer.config["model_name"] = request.param
    TestNevergradOptimizer.config["num_workers"] = num_workers


# Test suite for algorithms. You may reimplement some of the tests to adapt them to your algorithm
# Full documentation is available at https://orion.readthedocs.io/en/stable/code/testing/algo.html
# Look for algorithms tests in https://github.com/Epistimio/orion/blob/master/tests/unittests/algo
# for examples of customized tests.
class TestNevergradOptimizer(BaseAlgoTests):
    """Test suite for algorithm NevergradOptimizer"""

    algo_name = "nevergradoptimizer"
    config = {
        "seed": 1234,  # Because this is so random
        "budget": 100,
    }

    @phase
    def test_normal_data(self, mocker, num, attr):
        """Test that algorithm supports normal dimensions"""
        self.assert_dim_type_supported(mocker, num, attr, {"x": "normal(2, 5)"})

    def get_num(self, num):
        return min(num, 5)

    def test_optimize_branin(self):
        """Test that algorithm optimizes somehow (this is on-par with random search)"""
        MAX_TRIALS = 20  # pylint: disable=invalid-name
        task = Branin()
        space = self.create_space(task.get_search_space())
        algo = self.create_algo(config={}, space=space)
        algo.algorithm.max_trials = MAX_TRIALS
        safe_guard = 0
        trials = []
        objectives = []
        while trials or not algo.is_done:
            if safe_guard >= MAX_TRIALS:
                break

            if not trials:
                remaining = MAX_TRIALS - len(objectives)
                trials = algo.suggest(self.get_num(remaining))

            trial = trials.pop(0)
            results = task(trial.params["x"])
            objectives.append(results[0]["value"])
            backward.algo_observe(algo, [trial], [dict(objective=objectives[-1])])
            safe_guard += 1

        assert algo.is_done
        assert min(objectives) <= 10


# You may add other phases for test.
# See https://github.com/Epistimio/orion.algo.skopt/blob/master/tests/integration_test.py
# for an example where two phases are registered, one for the initial random step, and
# another for the optimization step with a Gaussian Process.
TestNevergradOptimizer.set_phases([("random", 0, "space.sample")])
