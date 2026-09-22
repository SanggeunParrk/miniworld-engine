from dual_experiment import Experiment
class Plan(Experiment):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,source="dual_stage32",**kwargs)
