"""Worked independent SFT variant. Read only after your first attempt."""
from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('variant_sft',ROOT/'labs/M12/sft.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
s.TRAIN += [
 ('I changed the cache and tested only one request.','LIMIT: Explain measurement limits.'),
 ('The API felt faster but I did not measure latency.','LIMIT: Explain measurement limits.'),
 ('I compared two runs on different machines.','LIMIT: Explain measurement limits.'),
 ('I measured response time once after deployment.','LIMIT: Explain measurement limits.'),
 ('I deleted the slow measurements from my report.','LIMIT: Explain measurement limits.'),
 ('I changed the index but the input workload also changed.','LIMIT: Explain measurement limits.'),
]
s.CHECK += [
 ('I only recorded the fastest result out of ten requests.','LIMIT: Explain measurement limits.'),
 ('I compared latency with a different number of users.','LIMIT: Explain measurement limits.'),
 ('I measured one warm request and called it the general latency.','LIMIT: Explain measurement limits.'),
 ('My benchmark used a much smaller database after the change.','LIMIT: Explain measurement limits.'),
]
s.SYSTEM += ' If the measurement procedure is unreliable, ask to explain measurement limits.'
if __name__=='__main__':
    adapter=ROOT/'artifacts/M12/sft/adapter.pt'
    if not adapter.exists():raise FileNotFoundError('Run labs/M12/sft.py first')
    s.main(ROOT/'artifacts/M12/sft-variant',steps=36,comparison_adapter=adapter)
