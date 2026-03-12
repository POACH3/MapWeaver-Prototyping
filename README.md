# MapWeaver (prototyping code)

MapWeaver is a tool for creating tabletop RPG maps and managing and playing game sessions. It offers Game Masters (GMs) greater creative freedom and the ability to manage intricate gameplay by automatically converting static tabletop RPG map images (scans, screenshots, hand-drawn maps) into large, playable maps with virtual terrain. Enabled by the map’s virtual terrain, powerful features offer improved gameplay experience. Beyond the powerful tools to aid GMs, it provides a low-friction, highly flexible option to players for session connection whether in-person or remote.


![Python](https://img.shields.io/badge/python-3.11+-blue)
[![License](https://img.shields.io/badge/License-Personal%20Use%20Only-orange.svg)](LICENSE)
<!--[![Tests](https://img.shields.io/badge/tests-%20none-lightgrey)]()-->

<br>
<br>

## Overview

This repo holds code that is disposable and of limited scope. The intent is to play around with and better understand different image processing methods.

A primary goal of MapWeaver is to import user maps and assist them in the creation of virtual terrain. Key elements are the automatic detection of both the gameplay grid as well as terrain features (such as walls/buildings, trees, and water).

Ultimately, the characteristics assigned to each virtual terrain type will drive gameplay features like enforced player movement, fog of war (through 2D ray-tracing), cover from opponent attacks, and concealment.


## Project Status

### Current 
- Grid detection by clustering Hough lines (Hough transform and DBSCAN)
- Synthetic data generation (a labeled grid dataset)

### Planned
- Grid detection with convolutional neural network
- Terrain detection (semantic segmentation, convolutional neural network)

### Known Limitations
- Grid detection with Hough lines is very brittle

<br>
<br>

## Project Info
**Status:** Prototype code (this is disposable code of limited scope)  
**Author:** T. Stratton  
**Start Date:** 2-MAR-2026  
**License:** Non-Commercial, Personal Use Only – see [LICENSE](./LICENSE)  
**Language:** Python 3.11+ (tested on 3.11)   
**Topics:** image-processing, computer-vision