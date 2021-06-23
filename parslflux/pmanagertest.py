import json, flux

from flux.job import JobspecV1
from flux.core.inner import ffi, raw



f = flux.Flux("ssh://c26/tmp/flux-hzoxDb/0/local")

r = f.rpc (b"parslmanager.register.uri", {"jobid":"123", "workeruri": "456"})
