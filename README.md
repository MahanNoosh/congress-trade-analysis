# CSC111 Project 2: Congress Stock Trading Analysis 📊

This project analyzes congressional stock trading data and visualizes it using both **graph-based** and **binary search tree (BST)** representations.  
The program filters raw CSV data, computes win rates for representatives, builds a balanced BST, and creates a network graph of politicians and stock sectors.

---

## Features

- Filter raw transaction data (`all_transactions.csv`) to relevant columns.  
- Build a **Balanced BST** of representatives based on trading "win rates."  
- Render BST visualization with **Graphviz**.  
- Build a **network graph** connecting representatives to stock sectors.  
- Visualize the graph using **Plotly** with colors based on party affiliation and sector.

---

## Requirements

- Python 3.10+  
- Libraries:
  - `networkx`
  - `plotly`
  - `graphviz`
- Ensure Graphviz is installed and added to system PATH.

---

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/MahanNoosh/CSC111_Project2.git
    cd CSC111_Project2
    ```
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the program:
   ```bash
    python src/main.py
   ```
- The program automatically filters the raw data into House_data.csv.
- Generates and opens a BST visualization (bst_winrate.png).
- Creates and displays a network graph of stock trades and sectors.
  
---

Contributions, feedback, and improvements are welcome!
