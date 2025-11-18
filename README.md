# Learning-based A*: How Deep Learning Neural Networks Can Improve Heuristic Search Algorithms

## Authors

Short information about the repository authors:

* Rezchikov Daniil, 4th-year student in Mechatronics and Robotics (MIREA - Russian Technological University). Fields of interest: navigation systems in mobile robotics, machine/deep learning, multi-agent robotic systems, computer vision.
* Ivanov Alexandr, 4th-year student in Mechatronics and Robotics (MIREA - Russian Technological University). Fields of interest: UAVs, machine/deep learning, control theory, computer vision.

## Project Goals

During the project, the following tasks should be completed:

1. **Implement loss masking** by excluding pixels that correspond to obstacles, start points, or goal points.
2. **Evaluate the developed approach** (1) on classical search algorithms such as D*, D* Lite, and Theta*.
3. **Train neural networks** on datasets containing warehouse and city maps.
4. **Replace RESNET blocks** with EfficientNet blocks and compare performance.
5. **Replace the Transformer module** (2) with CCT, CVT, or ViT alternatives.
6. **Compare results using the metrics** from the original paper and additionally compare execution time of algorithms (or their combinations with neural networks) with the classical A* algorithm. The time-based metric is measured in percentage relative to classic A*.

## References

Three main research works relevant to the project:

* **Article 1:** Kirilenko, D., Andreychuk, A., Panov, A., & Yakovlev, K. (2023). TransPath: Learning Heuristics for Grid-Based Pathfinding via Transformers. Proceedings of the AAAI Conference on Artificial Intelligence, 37(10), 12436-12443. https://doi.org/10.1609/aaai.v37i10.26465.
* **Article 2:** Hassani, Ali & Walton, Steven & Shah, Nikhil & Abuduweili, Abulikemu & Li, Jiachen & Shi, Humphrey. (2021). Escaping the Big Data Paradigm with Compact Transformers. 10.48550/arXiv.2104.05704. 
* **Article 3:** Daniil Kirilenko, Anton Andreychuk, Aleksandr I. Panov, Konstantin Yakovlev, Generative models for grid-based and image-based pathfinding, Artificial Intelligence, Volume 338, 2025, 104238, ISSN 0004-3702, https://doi.org/10.1016/j.artint.2024.104238.

## Datasets

The project uses datasets consisting of different types of map environments:

* Maps of **warehouse layouts**, for example the ones described in the [warehouse research dataset](http://example.com/warehouse-dataset).
* Maps of **urban environments**, similar to those in the [city planning dataset](http://example.com/city-dataset).
* A small custom dataset assembled from the TransPath_data dataset for validating hypotheses and preliminary experiments on a regular laptop
