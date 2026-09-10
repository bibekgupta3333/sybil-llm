# One-off migration script — already applied to eda_veremi.ipynb (intro_sybil_attack
# cell is present). Re-running will insert a duplicate cell.
import json
import os

notebook_path = os.path.join(os.path.dirname(__file__), "eda_veremi.ipynb")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

intro_cell = {
 "cell_type": "markdown",
 "id": "intro_sybil_attack",
 "metadata": {},
 "source": [
  "## Introduction\n",
  "\n",
  "In Vehicular Ad-hoc Networks (VANETs), a **Sybil attack** occurs when a single malicious physical entity claims multiple fictitious identities (ghost vehicles). This creates the illusion of multiple vehicles on the road, allowing attackers to broadcast conflicting messages, disrupt traffic management, and create safety hazards (e.g., phantom traffic jams).\n",
  "\n",
  "**Problem Context:** Detecting these attacks is challenging because individual forged messages often appear locally plausible. The true malicious intent is typically hidden in temporal inconsistencies or unrealistic motion patterns over a sequence of messages. The EDA below explores these patterns to prepare data for foundation model training.\n",
  "\n",
  "### Understanding the Attack Surface\n",
  "<!-- TODO: add a real Sybil-attack concept diagram under proposal/Figures/ and reference it here -->"
 ]
}

# Insert as the second cell (after the title) or first cell. Let's insert as the second cell.
nb["cells"].insert(1, intro_cell)

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
