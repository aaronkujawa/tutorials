#!/usr/bin/env python-real

import os
import sys
import subprocess
import re
from pathlib import Path

if __name__ == '__main__':

  # define progress bar steps as fraction of 1
  docker_command_received = 0.1
  first_output_from_docker = 0.2
  tqdm_start = docker_command_received+first_output_from_docker
  tqdm_end = 0.9

  command=sys.argv[1]
  print("command = ", command)
  cmd=command.split()

  # send progress bar start line to stdout
  print("""<filter-start><filter-name>TestFilter</filter-name><filter-comment>ibid</filter-comment></filter-start>""")
  sys.stdout.flush()

  print("""<filter-progress>{}</filter-progress>""".format(docker_command_received))
  sys.stdout.flush()
  
  def execute(cmd):

    # create startup environment for subprocess (to run python outside slicer)
    slicer_path = Path(os.environ["SLICER_HOME"]).resolve()
    PATH_without_slicer = os.pathsep.join([p for p in os.environ["PATH"].split(os.pathsep) if not slicer_path in Path(p).parents])

    startupEnv = {}
    try:
      startupEnv["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    except:
      print(["SYSTEMROOT environment variable does not exist..."])

    try:
      startupEnv["USERPROFILE"] = os.environ["USERPROFILE"]
    except:
      print(["USERPROFILE environment variable does not exist..."])
    
    startupEnv['PATH'] = PATH_without_slicer

    popen = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True, env=startupEnv)
    for line_idx, stderr_line in enumerate(iter(popen.stderr.readline, "")):
      print(stderr_line)
      if line_idx==0:
        print("""<filter-progress>{}</filter-progress>""".format(first_output_from_docker))
        sys.stdout.flush()

      # check if tqdm sent a progress line from docker (via stderr)
      match = re.findall(pattern, stderr_line)
      if match:
        # send progress bar update to stdout
        tqdm_progress = float(match[0])/100.0
        relative_progress = (tqdm_end - tqdm_start)*tqdm_progress + tqdm_start
        print("""<filter-progress>{}</filter-progress>""".format(relative_progress))
        sys.stdout.flush()

      yield stderr_line
    popen.stderr.close()
    return_code = popen.wait()

    # send progress bar completed to stdout
    #print("""<filter-end><filter-name>TestFilter</filter-name><filter-time>10</filter-time></filter-end>""")
    #sys.stdout.flush()

    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

  pattern = r"[\s]*([\d]+)%\|"

  print("--------------------------------", cmd)
  for line in execute(cmd):
    pass
