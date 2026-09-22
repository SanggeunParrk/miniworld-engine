from dual_experiment import Experiment
class Plan(Experiment):
    def __init__(self, d, dy, saved, count=132, part=2):
        super().__init__(d, dy, saved, count=count, part=part, source='dual_stats_soa')
