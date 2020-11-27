#!/bin/bash
python -m sc2arena.run_mp_game \
  --map 4MarineA \
  --player1 sc2arena.agents.simple_agent.AtkWeakestAgent \
  --screen_resolution 64 \
  --step_mul 8 \
  --novisualize \
  --agent_interface_format "rgb" \
  --sleep_time_per_step 0
