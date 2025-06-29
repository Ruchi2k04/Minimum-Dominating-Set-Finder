# Minimum-Dominating-Set-Finder

## Overview

This project solves the Minimum Dominating Set (MDS) problem using quantum computing. The MDS problem involves finding the smallest set of nodes in a graph such that all other nodes are either in the set or directly connected to it. We used Grover’s algorithm with the Qiskit library to process the graph and find the most likely solutions. The web app lets users input a graph, processes it using a quantum circuit, and displays the most probable solutions as a histogram. The goal is to show how quantum computing can help solve complex problems in a faster and way when compared to traditional methods.

## Features

- **Graph Input**: Accepts user-defined number of vertices and edges.
- **Quantum Algorithm**: Implements Grover’s search algorithm to find valid minimum dominating sets.
- **Output**: Displays a histogram that shows which solutions are most likely, based on quantum probability.
- **User-friendly Interface**: Clean Flask-based interface for smooth interaction.
