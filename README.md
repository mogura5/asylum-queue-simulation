# Asylum Queue Simulation

A discrete-event simulation of the U.S. asylum processing system modeled as a two-stage tandem queue. This project evaluates how different scheduling algorithms (FIFO, LIFO, Priority-based) affect system performance under normal and crisis conditions.

## Features

- Two-stage queue model (Asylum Officer → Immigration Judge)
- Configurable scheduling algorithms with aging mechanisms
- Case attributes based on empirical TRAC EOIR data
- Multiple court cluster archetypes (high/low volume, grant rates)
- Comprehensive metrics tracking (wait times, backlog, throughput)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd asylum-queue-simulation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt