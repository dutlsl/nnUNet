![](_page_0_Picture_0.jpeg)

![](_page_0_Picture_1.jpeg)

# <span id="page-0-0"></span>Mamba Learns in Context: Structure-Aware Domain Generalization for Multi-Task Point Cloud Understanding

Jincen Jiang<sup>1</sup> Qianyu Zhou<sup>2</sup> † Yuhang Li<sup>3</sup> Kui Su<sup>4</sup> Meili Wang<sup>5</sup> Jian Chang<sup>1</sup> Jian Jun Zhang<sup>1</sup> Xuequan Lu<sup>3</sup> †

<sup>1</sup>Bournemouth University <sup>2</sup> Jilin University <sup>3</sup>The University of Western Australia <sup>4</sup>Hangzhou City University <sup>5</sup>Northwest A&F University

jiangj@bournemouth.ac.uk zhouqianyu@jlu.edu.cn bruce.lu@uwa.edu.au

## Abstract

*While recent Transformer and Mamba architectures have advanced point cloud representation learning, they are typically developed for single-task or single-domain settings. Directly applying them to multi-task domain generalization (DG) leads to degraded performance. Transformers effectively model global dependencies but suffer from quadratic attention cost and lack explicit structural ordering, whereas Mamba offers linear-time recurrence yet often depends on coordinate-driven serialization, which is sensitive to viewpoint changes and missing regions, causing structural drift and unstable sequential modeling. In this paper, we propose Structure-Aware Domain Generalization (SADG), a Mamba-based In-Context Learning framework that preserves structural hierarchy across domains and tasks. We design structure-aware serialization (SAS) that generates transformation-invariant sequences using centroid-based topology and geodesic curvature continuity. We further devise hierarchical domain-aware modeling (HDM) that stabilizes cross-domain reasoning by consolidating intradomain structure and fusing inter-domain relations. At test time, we introduce a lightweight spectral graph alignment (SGA) that shifts target features toward source prototypes in the spectral domain without updating model parameters, ensuring structure-preserving test-time feature shifting. In addition, we introduce MP3DObject, a real-scan object dataset for multi-task DG evaluation. Comprehensive experiments demonstrate that the proposed approach improves structural fidelity and consistently outperforms state-of-the-art methods across multiple tasks including reconstruction, denoising, and registration. Our source code is available at: https://github.com/Jinec98/SADG.*

# 1. Introduction

Understanding 3D point clouds is essential for perception [\[15,](#page-8-0) [56,](#page-10-0) [63,](#page-10-1) [64,](#page-10-2) [67,](#page-10-3) [82,](#page-11-0) [97\]](#page-12-0), reconstruction [\[3,](#page-8-1) [23–](#page-9-0) [25,](#page-9-1) [44,](#page-9-2) [47\]](#page-9-3), and interaction [\[13,](#page-8-2) [14,](#page-8-3) [35,](#page-9-4) [66,](#page-10-4) [100\]](#page-12-1) in realworld systems. Most recent advances build on Transformerbased architectures [\[1,](#page-8-4) [17,](#page-8-5) [33,](#page-9-5) [60,](#page-10-5) [84,](#page-11-1) [85,](#page-11-2) [98\]](#page-12-2) to capture long-range dependencies via self-attention, while recent work explores State-Space Models such as Mamba [\[2,](#page-8-6) [18,](#page-8-7) [43,](#page-9-6) [53,](#page-10-6) [93,](#page-11-3) [95\]](#page-11-4) for linear-time sequence modeling. Although these architectures achieve strong performance on standard benchmarks, they are typically designed for single-task learning and struggle to generalize across unseen domains or handle multiple point cloud understanding tasks such as reconstruction, denoising, and registration. In practice, sensor variation, viewpoint differences, and scene incompleteness significantly challenge their generalization ability.

To address multi-task domain generalization (DG) for point clouds, DG-PIC [\[28\]](#page-9-7) is the first work to explore this direction. It conditions target samples on sourcedomain prompts, enabling a unified In-Context Learning model to perform multiple tasks. However, DG-PIC relies on Transformers, inheriting high computational complexity and lacking explicit token ordering. A natural replacement is Mamba, but this raises the following challenges: existing Mamba-based methods often depend on coordinate-driven serialization (*e.g.,* Axis Scanning and Hilbert Curves) for global alignment, which are sensitive to viewpoint changes and missing surfaces. Such serializations often break hierarchical object structure, causing unstable state propagation in Mamba and degrading generalization to unseen domains.

The core difficulty lies in multi-task learning: reconstruction, denoising, and registration all rely on preserving structural hierarchy, including global topology (part-whole spatial organization) and local geometric continuity (surface smoothness and curvature). Under domain shifts such as noise, occlusion, and pose variation, coordinate-driven se-

<sup>†</sup>Corresponding authors.

<span id="page-1-0"></span>rializations can distort sequence-local neighborhoods and disrupt intrinsic topological and geometric structure, making Mamba's recurrence fragile and feature-based domain alignment cannot ensure structural consistency. Therefore, robust multi-task DG requires explicitly encoding structure-aware token organization, enabling stable sequential modeling and structurally grounded alignment across domains.

In this paper, we propose Structure-Aware Domain Generalization (SADG), the first Mamba-based In-Context Learning framework for multi-task point cloud domain generalization. Our core idea is to explicitly serialize and align intrinsic geometric structure across domains and tasks. Our SADG consists of three key components. Firstly, we introduce a **Structure-Aware Serialization** (SAS) strategy based on two intrinsic spectra: a Centroid Distance Spectrum that preserves global topology and a Geodesic Curvature Spectrum that captures surface continuity, producing transformation-invariant and structure-consistent token sequences for Mamba that allow recurrent state propagation to reflect the underlying object hierarchy. Secondly, we design Hierarchical Domain-Aware Modeling (HDM) that first consolidates intra-domain structure and then performs inter-domain relational fusion within a unified sequence. Finally, a lightweight test-time Spectral Graph **Alignment (SGA)** module conducts graph spectral shifting to match target features to source prototypes without model updates, ensuring structure-preserving generalization.

Existing multi-domain multi-task point cloud benchmarks are limited in scale and real-scene variability, particularly in pose, occlusion, and sensor noise. To address this, we introduce **MP3DObject**, a new dataset of object-level real scans from Matterport3D [5] indoor scans, offering a strong testbed for synthetic-to-real generalization and a valuable resource for broader 3D understanding tasks. Extensive experiments on multiple datasets including this one show our method achieves state-of-the-art results.

Our main contributions are as follows:

- We identify the structural drift challenge in multi-task point cloud DG and present a Structure-Aware Domain Generalization framework that jointly preserves global topology and local geometry across domains and tasks.
- We propose structure-aware serialization for topology and curvature ordering, hierarchical domain-aware modeling for stable cross-domain reasoning, and spectral alignment for test-time structure-preserving shifting.
- We introduce MP3DObject, a new object-level dataset derived from Matterport3D, providing diverse real-world scans and a challenging benchmark for evaluating generalization from synthetic to real domains.

#### 2. Related Work

**Point Cloud Understanding.** Pioneered by PointNet [63] and PointNet++ [64], point-based methods [8, 26, 29, 32,

33, 37, 40, 45, 54, 74, 77, 83, 85, 90] directly learned permutation-invariant features from unordered point sets, while voxel-based [9, 15, 55, 62, 71], graph-based [27, 69, 77], and projection-based [1, 10, 38, 56, 81, 82, 97] captured local geometry. Transformer-based models [1, 17, 33, 41, 60, 84, 85, 98] achieved strong global reasoning but suffer from quadratic complexity and weak structural continuity. Recent Mamba-based models [2, 18, 42, 43, 53, 93, 95] enable efficient sequence modeling, but depend on coordinatebased serialization, making them sensitive to rotations and incomplete regions. Besides, they are designed for singletask learning and struggle to generalize to unseen domains. Point Cloud Domain Generalization (DG) seeks models that perform well on unseen domains without target data [50-52, 91, 101, 102]. Early DG works emphasize adversarial [65, 89], contrastive [46, 79, 80, 86] or augmentation-based [21, 31, 34, 87, 99] and consistencybased alignment [21, 30, 61], but neglect multi-task learning. Mamba-based DG frameworks [49, 91] improve efficiency but still rely on coordinate serialization. Recently, DG-PIC [28] is the first work that unifies multiple 3D tasks via Transformer prompts but remains computationally expensive and order-agnostic. Nevertheless, all these methods neglect learning the structure-preserving representations that remain invariant to domain and task variations.

Structure Modeling in Point Clouds has recently been explored to preserve the structural hierarchy of point clouds. Some researchers design structure-aware operators [7, 20, 36, 72, 76, 92] to encode local topology or surface continuity. Others construct graph-based representations [19, 22, 48, 68, 73, 75, 78, 96] to capture neighborhood relations. In addition, studies [4, 12, 39, 57–59, 70, 88] further explore hyperbolic geometry to embed hierarchical or tree-like structures in non-Euclidean space. However, these methods focus on single-domain or single-task structure modeling, but overlook the shared inherent structure across domains and tasks. In contrast, our approach is the first *structure-aware domain generalization* framework for multi-task point cloud understanding, explicitly preserving global topology and local geometry under domain shifts.

#### 3. Methodology

#### 3.1. Problem Setting and Overview

We study domain generalization for point cloud understanding in a multi-domain, multi-task setting, following DG-PIC [28]. Let  $\{D_s^k\}_{k=1}^K$  denote K source domains and  $D_t$  an unseen target domain. The model is trained on  $\{D_s^k\}$  and generalizes to  $D_t$  at test time without parameter updates. DG-PIC is formulated with In-Context Learning (ICL): given a point cloud, Farthest Point Sampling (FPS) and k-Nearest Neighbor (KNN) grouping produce patch tokens  $\mathcal{T} = \{t_i\}_{i=1}^N$ . A Transformer-based masked autoencoder

<span id="page-2-1"></span><span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Figure 1. Overview of our SADG. (a) During training, point clouds from multiple source domains are partitioned into local patches and serialized into structure-aware sequences using the Centroid Distance Spectrum (CDS) and Geodesic Curvature Spectrum (GCS), which preserve global topology and local geometric continuity. The serialized sequences are then processed by Mamba blocks under the Hierarchical Domain-Aware Modeling (HDM) mechanism, which stabilizes intra-domain structure and fuses inter-domain relations. (b) At test time, the Spectral Graph Alignment (SGA) performs structure-aware shifting that guides target features toward source prototypes in the spectral domain without updating model parameters, enabling robust generalization to unseen target domains across multiple tasks.

reconstructs query tokens from prompts, enabling a unified architecture for multiple tasks (*i.e.*, reconstruction, denoising, and registration) under a DG paradigm. However, this has drawbacks: (1) quadratic self-attention complexity limits scalability; (2) tokenization lacks explicit ordering, hindering sequential and structural consistency. Consequently, intrinsic geometric and topological cues are not fully captured, weakening generalization under domain shifts.

Motivated by this, we develop a Structure-Aware Domain Generalization (SADG) framework for point clouds. To our knowledge, this is the first work to introduce Mamba into ICL for domain-generalized multi-task point cloud understanding with structural consistency. Figure 1 shows that SADG first serializes unordered tokens into transformation-invariant sequences preserving topological and geometric relations, followed by a hierarchical domain-aware mechanism that captures intra-domain structure and inter-domain consistency. In testing, a spectral graph alignment module treats serialized target features as graph signals and aligns them with source prototypes in the spectral domain. These components preserve structural continuity, enhance sequential stability, and improve generalization across domains.

#### 3.2. Structure-Aware Serialization with Mamba

To overcome Transformer inefficiency and unordered tokenization, we adopt Mamba [16] as the sequential backbone for point cloud ICL. However, Figure 2 shows that Mamba is inherently order-sensitive: without a stable sequence, its recurrent updates become fragile and fail to capture structural relationships across domains and tasks. Thus, instead of coordinate-based token ordering, we introduce structure-aware serialization, which constructs intrinsic graph-based sequences encoding topological layout and geometric continuity, providing efficient and consistent inputs for Mamba. Notation. Following DG-PIC [28], FPS and KNN grouping produce N local patches (tokens)  $\mathcal{T} = \{t_i\}_{i=1}^N$  with centers  $u_i \in \mathbb{R}^3$  and features  $x_i \in \mathbb{R}^d$ . We construct a token graph  $\mathcal{G} = (\mathcal{V}, \mathcal{E}, w)$ , where  $\mathcal{V} = \{1, \dots, N\}$  indexes tokens and w(i,j) denotes the affinity between tokens  $t_i$  and  $t_j$ . A serialization is defined as a permutation:

$$\pi: \{\mathcal{V}_1, \dots, \mathcal{V}_N\} \xrightarrow{\text{serialize}} \{\mathcal{V}_{\pi(1)}, \dots, \mathcal{V}_{\pi(N)}\}, \quad (1)$$

which reorders tokens into a sequence:

$$X_{\pi} = [x_{\pi(1)}, \dots, x_{\pi(N)}],$$
 (2)

and is then processed by Mamba:

$$Z = \text{Mamba}(X_{\pi}) = [z_{\pi(1)}, \dots, z_{\pi(N)}].$$
 (3)

Different choices of w(i, j) yield different structure-aware serialization spectra, which we will introduce below.

<span id="page-3-1"></span><span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Figure 2. Comparison of different serialization strategies. The proposed CDS and GCS maintain transformation invariance and structural consistency across unaligned real-scan objects, providing a stable foundation for domain-generalized 3D understanding.

Centroid Distance Spectrum (CDS). To model the structural layout of unordered tokens, we establish a topology-aware serialization based on intrinsic spatial relationships. Given token centers  $\{u_i\}_{i=1}^N$ , we compute the point cloudwise global centroid:  $c=\frac{1}{N}\sum_{i=1}^N u_i$ . A naive choice would sort tokens directly by their distances to c, i.e.,  $d_i=\|u_i-c\|_2$ , which indeed provides a global measure but neglects local spatial continuity. Such sorting often causes abrupt transitions between spatially distant tokens, disrupting the topological smoothness essential for stable sequential modeling (see Figure 4 in Ablation). To preserve both local continuity and global coverage, we construct a token graph  $\mathcal{G}_{CDS}=(\mathcal{V},\mathcal{E},w_{CDS})$  with affinity:

$$w_{CDS}(i,j) = \exp\left(-\frac{\|u_i - u_j\|_2^2}{\sigma^2}\right),$$
 (4)

which softly connects nearby tokens and suppresses remote ones, thereby preserving local geometric coherence during traversal. Starting from the centroid-nearest token  $t_r = \arg\min_i \|u_i - c\|_2$ , we perform a Breadth-First Search (BFS) over  $\mathcal{G}_{CDS}$  to establish the serialization order. At each step, the current node expands to unvisited neighbors ranked by  $w_{CDS}(i,j)$ , ensuring that spatially adjacent tokens are explored in a locally smooth manner. This traversal continues until all tokens are visited, yielding a topology-consistent permutation  $\pi_{CDS}$  and the corresponding ordered sequence  $X_{\pi_{CDS}}$ . This serialization balances global coverage and local continuity, forming a coherent sequence encoding coarse-to-fine topological information of the point cloud for stable sequential modeling within Mamba.

Geodesic Curvature Spectrum (GCS). Beyond topology captured by CDS, GCS aims to encode intrinsic surface geometry through curvature-guided diffusion in a geodesic graph. General explicit curvature estimation relies on normals or dense sampling, which are fragile under noise, missing regions, and the domain gap between synthetic and

real scans (as shown in Figure 4). To overcome this, we formulate curvature implicitly through a heat diffusion process on the geodesic graph, providing a stable and intrinsic representation of local surface geometry.

Since Euclidean distances fail to reflect the intrinsic continuity of curved surfaces, we compute geodesic distances between tokens  $t_i$  and  $t_j$  as the shortest paths along a local KNN adjacency graph on token centers  $\{u_i\}_{i=1}^N$ :

$$d_{\text{geo}}(i,j) = \min_{\mathcal{P}_{ij}} \sum_{(p,q) \in \mathcal{P}_{ij}} ||u_p - u_q||_2,$$
 (5)

where  $\mathcal{P}_{ij}$  denotes the shortest valid path connecting  $t_i$  and  $t_j$ . This formulation follows manifold connectivity and preserves surface-aware consistency across complex geometric regions. To this end, we further define a curvature-guided heat diffusion using the Laplace–Beltrami operator  $\Delta$  on this geodesic graph. The diffusion equation  $\frac{\partial h(t)}{\partial t} = -\Delta h(t)$  implicitly captures local curvature behavior, where highly curved regions dissipate heat faster while flatter regions retain heat longer. The corresponding heat kernel between tokens  $t_i$  and  $t_j$  is expressed as:

$$K_{\tau}(i,j) = \sum_{k=1}^{N} e^{-\lambda_k \tau} \,\phi_k(i)\phi_k(j),\tag{6}$$

where  $\{\lambda_k,\phi_k\}$  are eigenvalues and eigenfunctions of  $\Delta$ . The diagonal term  $K_{\tau}(i,i)$  measures self-diffusion at node i, intrinsically encoding local curvature through diffusion dynamics. Sampling across multiple diffusion scales  $\{\tau_s\}_{s=1}^S$  yields a multi-scale curvature descriptor:

$$h_i = [K_{\tau_1}(i,i), K_{\tau_2}(i,i), \dots, K_{\tau_S}(i,i)]. \tag{7}$$

We then define curvature-based affinity between tokens as:

$$w_{GCS}(i,j) = \exp\left(-\frac{\|h_i - h_j\|_2^2}{\gamma^2}\right),$$
 (8)

and construct the token graph  $\mathcal{G}_{GCS} = (\mathcal{V}, \mathcal{E}, w_{GCS})$ . Serialization begins from the lowest-curvature token  $t_r = \arg\min_i \|h_i\|_2$  and proceeds in ascending curvature order, yielding a permutation  $\pi_{GCS}$  and the corresponding sequence  $X_{\pi_{GCS}}$ , which preserves geometric smoothness and curvature coherence across neighboring patches. This diffusion-based formulation encodes curvature intrinsically via heat propagation, providing stable geometric cues for low-quality data that strengthen Mamba's modeling. Together, CDS and GCS provide a transformation-invariant, structure-aware serialization that preserves both topological layout and geometric continuity for sequential modeling.

**Unified Structure-Aware Sequence.** To enhance contextual modeling with linear-time efficiency, we perform bidirectional traversals on both spectra and concatenate:

$$X_{\text{seq}} = [X_{\pi_{\text{CDS}}}; X_{\text{rev}(\pi_{\text{CDS}})}; X_{\pi_{\text{GCS}}}; X_{\text{rev}(\pi_{\text{GCS}})}]. \quad (9)$$

<span id="page-4-1"></span><span id="page-4-0"></span>![](_page_4_Figure_0.jpeg)

Figure 3. Hierarchical Domain-Aware Modeling (HDM) cascades intra-domain structural modeling and inter-domain relational fusion.

This unified, structure-aware sequence expands Mamba's receptive field while maintaining topological and geometric continuity, allowing Mamba to exploit ordered dependencies without sacrificing efficiency.

#### 3.3. Hierarchical Domain-Aware Modeling

Mamba processes tokens along the serialized order, making it well-suited to preserving topological and geometric continuity. Given the serialized input  $X_{\pi} = [x_{\pi(1)}, ..., x_{\pi(N)}]$ , Mamba updates hidden states recurrently:

$$z_t = \text{Mamba}(x_{\pi(t)}, z_{t-1}) = g(Az_{t-1} + Bx_{\pi(t)} + b), (10)$$

where A,B are learnable transition matrices, b is bias, and  $g(\cdot)$  the gating function. This update enables linear-time modeling of local continuity and long-range structure.

However, unlike Transformer-based ICL [28] which uses prompt-query concatenation along the unordered tokens, Mamba is order-sensitive. In DG, the simple concatenation of tokens from different domains disrupts the sequential dynamics and weakens state propagation, leading to unstable cross-domain reasoning.

To address this, we design a **Hierarchical Domain-Aware Modeling (HDM)** mechanism that reorganizes serialized features in two cascading stages, as illustrated in Figure 3, enhancing both intra-domain structural modeling and inter-domain relational generalization.

**Intra-domain Structural Modeling (ISM).** Given serialized sequences from prompt and query domains  $\{X_{\text{seq}}^p, X_{\text{seq}}^q\}$ , we perform intra-domain modeling to preserve structural dependencies by processing two parallel domain-specific Mamba branches independently:

$$Z^p = \operatorname{Mamba}^p(X_{\operatorname{seq}}^p), \quad Z^q = \operatorname{Mamba}^q(X_{\operatorname{seq}}^q).$$
 (11)

This stage stabilizes intra-domain consistency, not only ensuring that stable topological and geometric patterns aggregate within each domain before any cross-domain interaction, but also preventing sequential discontinuities across domain boundaries.

**Inter-domain Relational Fusion (IRF).** After obtaining  $\{Z^p, Z^q\}$ , we perform inter-domain relational fusion to establish transferable correspondences across domains. Different from direct concatenation used in Transformer-based

ICL [28], we interleave tokens from prompt and query domains following their shared structural order  $\pi$ :

$$Z^{pq} = [z_{\pi(1)}^p, z_{\pi(1)}^q, z_{\pi(2)}^p, z_{\pi(2)}^q, \dots, z_{\pi(4N)}^p, z_{\pi(4N)}^q],$$
(12)

producing a unified, structurally aligned sequence subsequently processed by a shared Mamba:

$$Z^f = \operatorname{Mamba}^f(Z^{pq}), \tag{13}$$

which jointly models domain-specific and domain-shared dependencies. The interleaved sequence implicitly exchanges features between domains through recurrent propagation without attention-based matching, enhancing relational generalization and structural consistency while maintaining linear efficiency.

#### 3.4. Spectral Graph Alignment

At test time, the model parameters remain frozen, and the goal is to preserve structural consistency on unseen domains. We propose a lightweight **Spectral Graph Alignment (SGA)** performing structure-aware alignment in the spectral domain before Mamba processing. Without requiring weight updates, SGA conducts spectral shifting on the latent graphs from CDS and GCS, ensuring topology- and geometry-consistent representations under domain shifts.

For each serialization strategy  $*\in \{CDS, GCS\}$ , let  $X_{\pi_*}$  denote the serialized token sequence, which we treat as a graph signal on  $\mathcal{G}_* = (\mathcal{V}, \mathcal{E}, w_*)$ . Using the normalized Laplacian  $\mathbf{L}_* = \mathbf{D}_* - \mathbf{A}_*$ , the Graph Fourier Transform (GFT) projects the sequence into the spectral domain as  $\hat{X}_* = \Phi_*^\top X_{\pi_*}$ , where  $\Phi_*$  denotes the eigenvectors of  $\mathbf{L}_*$  serving as the structural frequency bases.

To guide the domain generalization, we derive two source prototypes,  $\hat{P}_{CDS}^s$  and  $\hat{P}_{GCS}^s$ , which are computed by averaging source domain features and projecting them onto the unified query-specific spectral basis:

$$\hat{P}_*^s = (\Phi_*^t)^\top \left( \frac{1}{N_s} \sum_{i=1}^{N_s} X_{\pi_*,i}^s \right). \tag{14}$$

These prototypes capture domain-level statistics as stable structural anchors for spectral alignment. During testing,

<span id="page-5-1"></span><span id="page-5-0"></span>

| Table 1. Comparison on the multi-domain and multi-task benchmark. All models are trained on four source domains and directly evaluated |
|----------------------------------------------------------------------------------------------------------------------------------------|
| on the remaining one. Evaluation metric: Chamfer Distance (CD, $\times 10^{-3}$ , lower is better).                                    |

| Method            | Setting | ModelNet |       |       | ShapeNet |       |       | ScanNet |       |       | ScanObjectNN |       |       | MP3DObject |       |       |
|-------------------|---------|----------|-------|-------|----------|-------|-------|---------|-------|-------|--------------|-------|-------|------------|-------|-------|
| Method            | Setting | Rec.     | Den.  | Reg.  | Rec.     | Den.  | Reg.  | Rec.    | Den.  | Reg.  | Rec.         | Den.  | Reg.  | Rec.       | Den.  | Reg.  |
| PointNet [63]     | General | 20.56    | 27.15 | 17.19 | 19.84    | 32.99 | 19.57 | 23.73   | 30.27 | 19.49 | 21.74        | 33.97 | 21.62 | 22.63      | 35.24 | 23.17 |
| DGCNN [77]        | General | 19.17    | 26.91 | 17.41 | 21.18    | 26.97 | 19.05 | 21.73   | 30.35 | 17.51 | 24.88        | 33.08 | 18.85 | 21.83      | 38.10 | 21.82 |
| PCT [17]          | General | 16.98    | 25.10 | 14.50 | 18.39    | 24.59 | 15.28 | 18.76   | 27.03 | 16.99 | 19.50        | 29.98 | 15.71 | 18.91      | 28.74 | 16.49 |
| Point-MAE [60]    | General | 14.77    | 21.53 | 13.42 | 16.82    | 23.91 | 14.36 | 16.16   | 24.54 | 16.79 | 19.27        | 28.69 | 15.74 | 20.39      | 27.28 | 17.13 |
| PointMamba [43]   | General | 16.47    | 23.13 | 13.65 | 16.33    | 25.24 | 14.96 | 15.61   | 22.43 | 14.30 | 17.16        | 25.44 | 17.64 | 20.16      | 27.08 | 17.40 |
| PointMixup [6]    | DG      | 17.62    | 29.24 | 16.07 | 18.01    | 26.91 | 17.22 | 18.86   | 30.20 | 19.24 | 21.17        | 30.58 | 21.37 | 22.54      | 30.99 | 18.42 |
| PointCutMix [94]  | DG      | 16.23    | 27.07 | 16.77 | 18.19    | 25.30 | 16.24 | 21.28   | 29.21 | 19.07 | 22.64        | 28.58 | 19.85 | 22.80      | 33.24 | 18.88 |
| PointDGMamba [91] | DG      | 14.39    | 19.37 | 12.44 | 14.56    | 24.05 | 14.22 | 14.67   | 23.10 | 12.97 | 18.19        | 27.07 | 14.51 | 17.99      | 26.82 | 14.66 |
| PIC [11]          | ICL     | 17.89    | 25.70 | 16.15 | 17.96    | 24.10 | 15.50 | 16.90   | 30.46 | 18.10 | 21.75        | 29.42 | 16.74 | 22.86      | 34.72 | 17.54 |
| DG-PIC [28]       | ICL+DG  | 6.84     | 9.40  | 5.01  | 8.02     | 9.81  | 7.41  | 5.21    | 9.71  | 5.10  | 4.52         | 12.74 | 4.17  | 5.91       | 10.40 | 5.64  |
| Vanilla Mamba ICL | ICL+DG  | 7.69     | 10.81 | 6.22  | 7.98     | 10.19 | 6.25  | 5.45    | 10.75 | 5.56  | 6.93         | 11.52 | 7.76  | 8.28       | 14.19 | 8.44  |
| Ours (SADG)       | ICL+DG  | 5.99     | 7.98  | 3.81  | 7.64     | 9.34  | 7.06  | 2.97    | 7.67  | 3.63  | 4.29         | 9.84  | 3.03  | 3.55       | 6.61  | 2.84  |

target spectral tokens  $\hat{X}_{*}^{t}$  align toward the prototypes:

$$\hat{X}_{*i}^{t} \leftarrow \alpha_{i} \, \hat{X}_{*i}^{t} + (1 - \alpha_{i})(\hat{P}_{*}^{s} - \hat{X}_{*i}^{t}), \tag{15}$$

where the adaptive coefficient  $\alpha_i$  is modulated by the cosine similarity between  $\hat{X}^t_{*,i}$  and  $\hat{P}^s_*$ , enforcing coherent alignment while avoiding over-correction in irregular regions. The aligned spectral features are then transformed back to the spatial domain via the inverse GFT as  $X^t_{\pi_*} = \Phi^t_* \hat{X}^t_*$ .

By leveraging the intrinsic graphs of CDS and GCS, this spectral process preserves topological and geometric consistency while explicitly mitigating domain discrepancies in the spatial space. Consequently, SGA performs test-time, structure-aware alignment without weight updates, providing a transformation-invariant foundation that enables the model to better generalize to unseen target domains.

#### 4. Experiments

#### 4.1. Benchmark and Implementation Details

**Multi-domain Multi-task Benchmark.** Following DG-PIC [28], we evaluate our method under a unified multi-domain, multi-task setting. The benchmark integrates five datasets with consistent category definitions across seven shared classes (*chair*, *table*, *sofa*, *bed*, *cabinet*, *shelf*, *monitor*), including two synthetic (*ModelNet40*, *ShapeNet*) and three real-scan domains (*ScanNet*, *ScanObjectNN*, and the newly introduced *MP3DObject*). All point clouds are sampled to 1,024 points and normalized within a unit sphere. Three representative tasks, *i.e.*, reconstruction, denoising, and registration, are learned jointly within a single unified model. This benchmark covers diverse domains and geometric conditions, offering a challenging yet comprehensive testbed for generalization across multiple tasks.

**MP3DObject.** To enable more realistic evaluation beyond DG-PIC [28], we construct *MP3DObject* from the large-scale indoor dataset Matterport3D [5] by extracting object-level instances and removing extremely incomplete samples. Each object is centered and normalized but not aligned

to a canonical orientation, introducing natural viewpoint and pose variations. This dataset contains 4,015 training samples and 1,003 testing samples, offering real-world variation while preserving complex geometry, making it a challenging benchmark for realistic point cloud understanding. Implementation. All experiments are implemented in Py-Torch with CUDA 11.8 and trained on a TITAN RTX GPU using AdamW with learning rate of  $1 \times 10^{-4}$ , cosine decay and batch size of 96 for 300 epochs. We set  $\sigma$  and  $\gamma$  to the medians of corresponding token graphs, yielding a robust and adaptive affinity scale. Following the leaveone-domain-out protocol, models are trained on four source domains and directly evaluated on the unseen target without parameter updates. We use the Chamfer Distance (CD) metric to measure geometric consistency between predictions and ground truth across all tasks.

### 4.2. Main Results

General-purpose Point Cloud Baselines. We compare our method with approaches for general point cloud learning, including PointNet [63], DGCNN [77], PCT [17], Point-MAE [60], and PointMamba [43]. All baselines are retrained under the same multi-domain, multi-task protocol, training on four sources and evaluating on the held-out target. While competitive in domain-specific or single-task settings, these models degrade in our unified benchmark (Table 1), with PointNet dropping sharply on real scans such as MP3DObject. Even modern backbones like Point-MAE and PointMamba still fail to generalize, e.g., Point-Mamba reaches 20.16/27.08/17.40 on MP3DObject, indicating sensitivity to viewpoint changes and incompleteness. These results highlight that without structure-aware design or domain modeling, general models fail to generalize across heterogeneous domains.

**Domain Generalization Methods.** DG approaches enhance robustness by diversifying sources or learning domain-invariant representations. PointMixup [6] and PointCutMix [94] mix cross-domain samples to encour-

<span id="page-6-1"></span>

<span id="page-6-2"></span>

| Variant                            | Reconstruction | Denoising | Registration |  |  |  |  |  |
|------------------------------------|----------------|-----------|--------------|--|--|--|--|--|
| Serialization Variants             |                |           |              |  |  |  |  |  |
| Z-order Scanning                   | 7.32           | 12.47     | 6.29         |  |  |  |  |  |
| Hilbert Curve                      | 6.23           | 11.13     | 7.68         |  |  |  |  |  |
| w/o CDS                            | 4.75           | 10.69     | 6.17         |  |  |  |  |  |
| w/o GCS                            | 5.82           | 8.92      | 4.43         |  |  |  |  |  |
| Ours (full SAS)                    | 3.55           | 6.61      | 2.84         |  |  |  |  |  |
| Hierarchical Domain-Aware Modeling |                |           |              |  |  |  |  |  |
| w/o ISM                            | 5.41           | 11.37     | 7.51         |  |  |  |  |  |
| w/o IRF                            | 6.92           | 9.12      | 7.57         |  |  |  |  |  |
| Ours (full HDM)                    | 3.55           | 6.61      | 2.84         |  |  |  |  |  |

age feature interpolation, while PointDGMamba [\[91\]](#page-11-10) leverages Mamba for improved domain consistency. Although effective under standard DG setups, their performance drops sharply in our multi-domain and multi-task benchmark, where models must simultaneously handle heterogeneous domains and multiple objectives. For instance, PointMixup obtains CD errors 18.86/30.20/19.24 on Scan-Net, and PointCutMix reaches 22.80/33.24/18.88 on MP3DObject, indicating persistent inconsistencies across domains and tasks. In contrast, our SADG achieves CD errors 2.97/7.67/3.63 on ScanNet and 3.55/6.61/2.84 on MP3DObject, substantially outperforming across all tasks. These results show that structure-aware serialization, hierarchical Mamba reasoning, and spectral graph alignment jointly enable coherent topology-geometry modeling across domains, yielding more stable and transferable features under distribution shifts.

In-context Learning Methods. We further compare with ICL-based frameworks that unify multiple tasks via prompt conditioning. PIC [\[11\]](#page-8-21) and DG-PIC [\[28\]](#page-9-7) demonstrate the benefit of prompt-query design, yet their Transformer backbones and unsorted token sequences hinder efficiency and stability. DG-PIC improves performance (*e.g.*, 6.84/9.40/5.01 CD on ModelNet) but remains constrained by quadratic attention and order-agnostic tokenization. To assess backbone influence, we implement a vanilla Mamba-based ICL by replacing DG-PIC's Transformer with Mamba blocks. Although more efficient, this variant shows unstable behavior (*e.g.*, 8.28/14.19/8.44 on MP3DObject) due to coordinate-based tokenization being highly sensitive to pose variation and incompleteness. With our Structure-Aware Serialization (SAS) and Hierarchical Domain-Aware Modeling (HDM), both stability and generalization improve markedly. As shown in Table [1,](#page-5-0) our model achieves 5.99/7.98/3.81 on ModelNet, 7.64/9.34/7.06 on ShapeNet, and the best results across all real-world scenarios like MP3DObject. These results confirm that SAS enables coherent topology-geometry reasoning in Mamba, while HDM and SGA further ensure robust generalization to unseen targets.

<span id="page-6-0"></span>![](_page_6_Picture_4.jpeg)

Figure 4. Comparison between naive serialization variants and our method, highlighting stable topology and geometry in 3D objects.

#### 4.3. Ablation Studies

We analyze each component using MP3DObject as targets, which contains complex geometries and unaligned poses. Serialization Variants. We compare several serialization strategies to validate our structure-aware design. As shown in Table [2,](#page-6-1) coordinate-based traversals such as Z-order and Hilbert yield higher CD errors (*e.g.,* 7.32/12.47/6.29 and 6.23/11.13/7.68), reflecting their sensitivity to orientation and lack of intrinsic structure embedding. Removing CDS or GCS further degrades performance (4.75/10.69/6.17 and 5.82/8.92/4.43), confirming that topology and geometry provide complementary cues for stable sequential modeling. Figure [4](#page-6-0) illustrates the naive options for our serialization. Top-left shows a simple centroid Euclidean distance ordering where the sequence repeatedly jumps across the surface, breaking local continuity. Bottom-left depicts curvature-based sorting, whose estimates are noisy on real scans and produce irregular surface ordering. Our design (*i.e.,* CDS and GCS) stably models topology and geometry relations, maintains strong structural coherence and domain robustness across multiple point cloud understanding tasks. Hierarchical Domain-Aware Modeling. HDM performs intra-domain structural modeling and inter-domain relational fusion within unified Mamba sequences. Due to Mamba's order sensitivity, direct concatenation of diverse domains disrupts state transitions. Table [2](#page-6-1) shows that removing ISM increases CD errors to 5.41/11.37/7.51, while removing IRF yields 6.92/9.12/7.57, both noticeably worse than our full HDM (3.55/6.61/2.84). These results verify that HDM stabilizes sequential updates and preserves consistent structural reasoning across domains.

#### 4.4. Visualization and Efficiency Analysis

Qualitative Results. Figure [5](#page-7-0) shows results for reconstruction, denoising, and registration. On synthetic targets such as ModelNet, our model restores fine details while preserving overall geometry. On real scans like MP3DObject, it achieves higher structural fidelity with clearer boundaries, fewer holes, and smoother surfaces. Notably, our method

<span id="page-7-2"></span><span id="page-7-0"></span>![](_page_7_Figure_0.jpeg)

Figure 5. Qualitative results on synthetic (ModelNet) and real-scan (MP3DObject) targets. Our method recovers both detailed geometry and smooth surfaces. For visualization, MP3DObject instances are rendered after aligning their non-canonical orientations.

<span id="page-7-1"></span>![](_page_7_Figure_2.jpeg)

Figure 6. t-SNE visualization of latent features. The Input plot shows encoder-only features, while the latter two depict representations after Mamba-based sequential modeling.

also better preserves thin structures and recovers missing surface regions. These qualitative improvements confirm that SAS provides robust topology and geometry cues across domains, particularly under noise and incompleteness. Moreover, the consistent reconstructed shapes across synthetic and real data indicates that our SADG framework maintains coherent structural reasoning even under severe viewpoint shifts and partial observations, further highlighting its advantage over existing baselines. More qualitative examples are provided in the supplementary material.

**t-SNE Visualization.** To evaluate domain alignment, we visualize Mamba features using t-SNE in Figure 6. Vanilla Mamba ICL produces largely domain-separated clusters, indicating weak transferability. Our method enables compact intra-domain clusters and stronger overlap between source

and unseen targets, showing improved cross-domain alignment without collapsing domain-specific structure, yielding more stable and generalizable embeddings with reduced feature fragmentation under severe distribution shifts, thus strengthening domain generalization.

Efficiency Analysis. We compare runtime and model complexity with DG-PIC [28]. Using the same input, we achieve 0.75s inference with 18.87M parameters and 14.89G FLOPs, versus DG-PIC's 0.94s, 27.57M, and 21.07G FLOPs, while delivering better results. This highlights the efficiency and scalability of our Mamba-based design, offering an improved efficiency-performance trade-off under the multi-domain and multi-task setting.

#### 5. Conclusion

We introduced a structure-aware domain generalization framework for multi-task point cloud understanding, which incorporates intrinsic topology and geometry into sequence modeling. By constructing the Centroid Distance Spectrum and Geodesic Curvature Spectrum, unordered tokens are serialized into transformation-invariant, structure-consistent sequences, enabling Mamba to model long-range dependencies efficiently and stably. To enhance cross-domain generalization, the proposed Hierarchical Domain-Aware Modeling performs intra-domain structural reasoning and inter-domain relational fusion within a unified sequential representation, while the Spectral Graph Alignment ensures structure-aware feature alignment at test time without parameter updates. Extensive experiments verify that our SADG achieves superior performance on various datasets.

# Acknowledgments

This work is supported by the National Natural Science Foundation of China (No. 62502178), Jilin University Special Program for Talent Development in Engineering Cluster Construction, the China Scholarship Council (No. 202306300023), and the Research and Development Fund of Bournemouth University.

# References

- <span id="page-8-4"></span>[1] Angelika Ando, Spyros Gidaris, Andrei Bursuc, Gilles Puy, Alexandre Boulch, and Renaud Marlet. Rangevit: Towards vision transformers for 3d semantic segmentation in autonomous driving. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 5240–5250, 2023. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-8-6"></span>[2] Ali Bahri, Moslem Yazdanpanah, Mehrdad Noori, Sahar Dastani, Milad Cheraghalikhani, Gustavo Adolfo Vargas Hakim, David Osowiechi, Farzad Beizaee, Ismail Ben Ayed, and Christian Desrosiers. Spectral informed mamba for robust point cloud processing. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 11799–11809, 2025. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-8-1"></span>[3] Matthew Berger, Andrea Tagliasacchi, Lee Seversky, Pierre Alliez, Joshua Levine, Andrei Sharf, and Claudio Silva. State of the art in surface reconstruction from point clouds. *Eurographics 2014-State of the Art Reports*, 1(1):161–185, 2014. [1](#page-0-0)
- <span id="page-8-17"></span>[4] Jian Bi, Qianliang Wu, Jianjun Qian, Lei Luo, and Jian Yang. Dual manifold regularization steered robust representation learning for point cloud analysis. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pages 1844– 1852, 2025. [2](#page-1-0)
- <span id="page-8-8"></span>[5] Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niessner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3d: Learning from rgb-d data in indoor environments. *International Conference on 3D Vision*, 2017. [2,](#page-1-0) [6,](#page-5-1) [4](#page-3-1)
- <span id="page-8-20"></span>[6] Yunlu Chen, Vincent Tao Hu, Efstratios Gavves, Thomas Mensink, Pascal Mettes, Pengwan Yang, and Cees GM Snoek. Pointmixup: Augmentation for point clouds. In *European Conference on Computer Vision*, pages 330–345. Springer, 2020. [6](#page-5-1)
- <span id="page-8-13"></span>[7] Zhihua Cheng and Xuejin Chen. Structure-aware point cloud completion. In *International Conference on Image and Graphics*, pages 174–185, 2023. [2](#page-1-0)
- <span id="page-8-9"></span>[8] Jaesung Choe, Chunghyun Park, Francois Rameau, Jaesik Park, and In So Kweon. Pointmixer: Mlp-mixer for point cloud understanding. In *European Conference on Computer Vision*, pages 620–640, 2022. [2](#page-1-0)
- <span id="page-8-10"></span>[9] Christopher Choy, JunYoung Gwak, and Silvio Savarese. 4d spatio-temporal convnets: Minkowski convolutional neural networks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 3075–3084, 2019. [2](#page-1-0)
- <span id="page-8-11"></span>[10] Tiago Cortinhal, George Tzelepis, and Eren Erdal Aksoy. Salsanext: Fast, uncertainty-aware semantic segmentation

- of lidar point clouds. In *International Symposium on Visual Computing*, pages 207–222. Springer, 2020. [2](#page-1-0)
- <span id="page-8-21"></span>[11] Zhongbin Fang, Xiangtai Li, Xia Li, Joachim M Buhmann, Chen Change Loy, and Mengyuan Liu. Explore in-context learning for 3d point cloud understanding. *Advances in Neural Information Processing Systems*, 36:42382–42395, 2023. [6,](#page-5-1) [7](#page-6-2)
- <span id="page-8-18"></span>[12] Yuan-Zhi Feng, Shing-Ho J Lin, Xuan Tang, Mu-Yu Wang, Jian-Zhang Zheng, Zi-Yao He, Zi-Yi Pang, Jian Yang, Ming-Song Chen, and Xian Wei. Hyperbolic prototype rectification for few-shot 3d point cloud classification. *Pattern Recognition*, 158:111042, 2025. [2](#page-1-0)
- <span id="page-8-2"></span>[13] Haoran Geng, Ziming Li, Yiran Geng, Jiayi Chen, Hao Dong, and He Wang. Partmanip: Learning cross-category generalizable part manipulation policy from point cloud observations. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 2978– 2988, 2023. [1](#page-0-0)
- <span id="page-8-3"></span>[14] Haoran Geng, Helin Xu, Chengyang Zhao, Chao Xu, Li Yi, Siyuan Huang, and He Wang. Gapartnet: Cross-category domain-generalizable object perception and manipulation via generalizable and actionable parts. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 7081–7091, 2023. [1](#page-0-0)
- <span id="page-8-0"></span>[15] Benjamin Graham, Martin Engelcke, and Laurens van der Maaten. 3d semantic segmentation with submanifold sparse convolutional networks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 9224–9232, 2018. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-8-19"></span>[16] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In *First Conference on Language Modeling*, 2024. [3](#page-2-1)
- <span id="page-8-5"></span>[17] Meng-Hao Guo, Jun-Xiong Cai, Zheng-Ning Liu, Tai-Jiang Mu, Ralph R Martin, and Shi-Min Hu. Pct: Point cloud transformer. *Computational Visual Media*, 7(2):187–199, 2021. [1,](#page-0-0) [2,](#page-1-0) [6](#page-5-1)
- <span id="page-8-7"></span>[18] Xu Han, Yuan Tang, Zhaoxuan Wang, and Xianzhi Li. Mamba3d: Enhancing local features for 3d point cloud analysis via state space model. In *Proceedings of the 32nd ACM International Conference on Multimedia*, pages 4995–5004, 2024. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-8-15"></span>[19] Fengda Hao, Jiaojiao Li, Rui Song, Yunsong Li, and Kailang Cao. Structure-aware graph convolution network for point cloud parsing. *IEEE Transactions on Multimedia*, 25:7025–7036, 2022. [2](#page-1-0)
- <span id="page-8-14"></span>[20] Chenhang He, Hui Zeng, Jianqiang Huang, Xian-Sheng Hua, and Lei Zhang. Structure aware single-stage 3d object detection from point cloud. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 11873–11882, 2020. [2](#page-1-0)
- <span id="page-8-12"></span>[21] Pei He, Lingling Li, Licheng Jiao, Ronghua Shang, Fang Liu, Shuang Wang, Xu Liu, and Wenping Ma. Domainaware category-level geometry learning segmentation for 3d point clouds. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 28324– 28333, 2025. [2](#page-1-0)
- <span id="page-8-16"></span>[22] Wei Hu, Xiang Gao, Gene Cheung, and Zongming Guo.

- Feature graph learning for 3d point cloud denoising. *IEEE Transactions on Signal Processing*, 68:2841–2856, 2020. [2](#page-1-0)
- <span id="page-9-0"></span>[23] Hui Huang, Dan Li, Hao Zhang, Uri Ascher, and Daniel Cohen-Or. Consolidation of unorganized point clouds for surface reconstruction. *ACM Transactions on Graphics*, 28 (5):1–7, 2009. [1](#page-0-0)
- [24] Zhangjin Huang, Yuxin Wen, Zihao Wang, Jinjuan Ren, and Kui Jia. Surface reconstruction from point clouds: A survey and a benchmark. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 46(12):9727–9748, 2024.
- <span id="page-9-1"></span>[25] Philipp Jenke, Michael Wand, Martin Bokeloh, Andreas Schilling, and Wolfgang Straßer. Bayesian point cloud reconstruction. In *Computer Graphics Forum*, pages 379– 388, 2006. [1](#page-0-0)
- <span id="page-9-8"></span>[26] Jincen Jiang, Xuequan Lu, Lizhi Zhao, Richard Dazeley, and Meili Wang. Masked autoencoders in 3d point cloud representation learning. *IEEE Transactions on Multimedia*, 27:820–831, 2023. [2](#page-1-0)
- <span id="page-9-14"></span>[27] Jincen Jiang, Lizhi Zhao, Xuequan Lu, Wei Hu, Imran Razzak, and Meili Wang. Dhgcn: dynamic hop graph convolution network for self-supervised point cloud learning. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pages 12883–12891, 2024. [2](#page-1-0)
- <span id="page-9-7"></span>[28] Jincen Jiang, Qianyu Zhou, Yuhang Li, Xuequan Lu, Meili Wang, Lizhuang Ma, Jian Chang, and Jian Jun Zhang. Dgpic: Domain generalized point-in-context learning for point cloud understanding. In *European Conference on Computer Vision*, pages 455–474. Springer, 2024. [1,](#page-0-0) [2,](#page-1-0) [3,](#page-2-1) [5,](#page-4-1) [6,](#page-5-1) [7,](#page-6-2) [8,](#page-7-2) [4](#page-3-1)
- <span id="page-9-9"></span>[29] Jincen Jiang, Qianyu Zhou, Yuhang Li, Xinkui Zhao, Meili Wang, Lizhuang Ma, Jian Chang, Jian J Zhang, and Xuequan Lu. Pcotta: Continual test-time adaptation for multitask point cloud understanding. *Advances in Neural Information Processing Systems*, 37:96229–96253, 2024. [2](#page-1-0)
- <span id="page-9-21"></span>[30] Hyeonseong Kim, Yoonsu Kang, Changgyoon Oh, and Kuk-Jin Yoon. Single domain generalization for lidar semantic segmentation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 17587–17598, 2023. [2](#page-1-0)
- <span id="page-9-19"></span>[31] Jaeyeul Kim, Jungwan Woo, Jeonghoon Kim, and Sunghoon Im. Rethinking lidar domain generalization: Single source as multiple density domains. In *European Conference on Computer Vision*, pages 310–327, 2024. [2](#page-1-0)
- <span id="page-9-10"></span>[32] Artem Komarichev, Zichun Zhong, and Jing Hua. A-cnn: Annularly convolutional neural networks on point clouds. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 7421–7430, 2019. [2](#page-1-0)
- <span id="page-9-5"></span>[33] Xin Lai, Jianhui Liu, Li Jiang, Liwei Wang, Hengshuang Zhao, Shu Liu, Xiaojuan Qi, and Jiaya Jia. Stratified transformer for 3d point cloud segmentation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 8500–8509, 2022. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-9-20"></span>[34] Alexander Lehner, Stefano Gasperini, Alvaro Marcos-Ramiro, Michael Schmidt, Mohammad-Ali Nikouei Mahani, Nassir Navab, Benjamin Busam, and Federico Tombari. 3d-vfield: Adversarial augmentation of point clouds for domain generalization in 3d object detection. In *Proceedings of the IEEE/CVF Conference on Computer Vi-*

- *sion and Pattern Recognition*, pages 17295–17304, 2022. [2](#page-1-0)
- <span id="page-9-4"></span>[35] Chengmeng Li, Junjie Wen, Yaxin Peng, Yan Peng, and Yichen Zhu. Pointvla: Injecting the 3d world into visionlanguage-action models. *IEEE Robotics and Automation Letters*, 11(3):2506–2513, 2026. [1](#page-0-0)
- <span id="page-9-22"></span>[36] Jiaxin Li, Ben M Chen, and Gim Hee Lee. So-net: Selforganizing network for point cloud analysis. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 9397–9406, 2018. [2](#page-1-0)
- <span id="page-9-11"></span>[37] Jianan Li, Jie Wang, and Tingfa Xu. Pointgl: A simple global-local framework for efficient point cloud analysis. *IEEE Transactions on Multimedia*, 26:6931–6942, 2024. [2](#page-1-0)
- <span id="page-9-15"></span>[38] Li Li, Hubert PH Shum, and Toby P Breckon. Rapid-seg: Range-aware pointwise distance distribution networks for 3d lidar segmentation. In *European Conference on Computer Vision*, pages 222–241. Springer, 2024. [2](#page-1-0)
- <span id="page-9-23"></span>[39] Wenrui Li, Zhe Yang, Wei Han, Hengyu Man, Xingtao Wang, and Xiaopeng Fan. Hyperbolic-constraint point cloud reconstruction from single rgb-d images. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pages 4959–4967, 2025. [2](#page-1-0)
- <span id="page-9-12"></span>[40] Yangyan Li, Rui Bu, Mingchao Sun, Wei Wu, Xinhan Di, and Baoquan Chen. Pointcnn: Convolution on xtransformed points. *Advances in Neural Information Processing Systems*, 31, 2018. [2](#page-1-0)
- <span id="page-9-16"></span>[41] Yinghui Li, Qianyu Zhou, Jingyu Gong, Ye Zhu, Richard Dazeley, Xinkui Zhao, and Xuequan Lu. Dapointr: Domain adaptive point transformer for point cloud completion. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pages 5066–5074, 2025. [2](#page-1-0)
- <span id="page-9-17"></span>[42] Yinghui Li, Qianyu Zhou, Di Shao, Hao Yang, Ye Zhu, Richard Dazeley, and Xuequan Lu. Dapointmamba: Domain adaptive point mamba for point cloud completion. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pages 6653–6661, 2026. [2](#page-1-0)
- <span id="page-9-6"></span>[43] Dingkang Liang, Xin Zhou, Wei Xu, Xingkui Zhu, Zhikang Zou, Xiaoqing Ye, Xiao Tan, and Xiang Bai. Pointmamba: A simple state space model for point cloud analysis. *Advances in Neural Information Processing Systems*, 37:32653–32677, 2024. [1,](#page-0-0) [2,](#page-1-0) [6](#page-5-1)
- <span id="page-9-2"></span>[44] Chen-Hsuan Lin, Chen Kong, and Simon Lucey. Learning efficient point cloud generation for dense 3d object reconstruction. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 2018. [1](#page-0-0)
- <span id="page-9-13"></span>[45] Haojia Lin, Xiawu Zheng, Lijiang Li, Fei Chao, Shanshan Wang, Yan Wang, Yonghong Tian, and Rongrong Ji. Meta architecture for point cloud analysis. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 17682–17691, 2023. [2](#page-1-0)
- <span id="page-9-18"></span>[46] Bangzhen Liu, Chenxi Zheng, Xuemiao Xu, Cheng Xu, Huaidong Zhang, and Shengfeng He. Rotation-adaptive point cloud domain generalization via intricate orientation learning. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 47(5):4232–4239, 2025. [2](#page-1-0)
- <span id="page-9-3"></span>[47] Fengqi Liu, Jingyu Gong, Qianyu Zhou, Xuequan Lu, Ran Yi, Yuan Xie, and Lizhuang Ma. Cloudmix: Dual mixup

- consistency for unpaired point cloud completion. *IEEE Transactions on Visualization and Computer Graphics*, 31 (4):2182–2195, 2024. [1](#page-0-0)
- <span id="page-10-18"></span>[48] Yifan Liu, Wuyang Li, Jie Liu, Hui Chen, and Yixuan Yuan. Grab-net: Graph-based boundary-aware network for medical point cloud segmentation. *IEEE Transactions on Medical Imaging*, 42(9):2776–2786, 2023. [2](#page-1-0)
- <span id="page-10-16"></span>[49] Shaocong Long, Qianyu Zhou, Xiangtai Li, Xuequan Lu, Chenhao Ying, Yuan Luo, Lizhuang Ma, and Shuicheng Yan. Dgmamba: Domain generalization via generalized state space model. In *Proceedings of the 32nd ACM International Conference on Multimedia*, pages 3607–3616, 2024. [2](#page-1-0)
- <span id="page-10-12"></span>[50] Shaocong Long, Qianyu Zhou, Chenhao Ying, Lizhuang Ma, and Yuan Luo. Rethinking domain generalization: Discriminability and generalizability. *IEEE Transactions on Circuits and Systems for Video Technology*, 34(11):11783– 11797, 2024. [2](#page-1-0)
- [51] Shaocong Long, Qianyu Zhou, Xikun Jiang, Chenhao Ying, Lizhuang Ma, and Yuan Luo. Domain generalization via discrete codebook learning. In *IEEE International Conference on Multimedia and Expo*, pages 1–6, 2025.
- <span id="page-10-13"></span>[52] Shaocong Long, Qianyu Zhou, Chenhao Ying, Lizhuang Ma, and Yuan Luo. Diverse target and contribution scheduling for domain generalization. *IEEE Transactions on Image Processing*, 34:4242–4257, 2025. [2](#page-1-0)
- <span id="page-10-6"></span>[53] Dening Lu, Kyle Gao, Jonathan Li, Dedong Zhang, and Linlin Xu. Exploring token serialization for mamba-based lidar point cloud segmentation. *IEEE Transactions on Geoscience and Remote Sensing*, 63:1–14, 2025. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-10-7"></span>[54] Xu Ma, Can Qin, Haoxuan You, Haoxi Ran, and Yun Fu. Rethinking network design and local geometry in point cloud: A simple residual mlp framework. In *International Conference on Learning Representations*, 2022. [2](#page-1-0)
- <span id="page-10-8"></span>[55] Daniel Maturana and Sebastian Scherer. Voxnet: A 3d convolutional neural network for real-time object recognition. In *2015 IEEE/RSJ International Conference on Intelligent Robots and Systems*, pages 922–928. IEEE, 2015. [2](#page-1-0)
- <span id="page-10-0"></span>[56] Andres Milioto, Ignacio Vizzo, Jens Behley, and Cyrill Stachniss. Rangenet++: Fast and accurate lidar semantic segmentation. In *IEEE/RSJ International Conference on Intelligent Robots and Systems*, pages 4213–4220, 2019. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-10-20"></span>[57] Antonio Montanaro, Diego Valsesia, and Enrico Magli. Rethinking the compositionality of point clouds through regularization in the hyperbolic space. *Advances in Neural Information Processing Systems*, 35:33741–33753, 2022. [2](#page-1-0)
- [58] Antonio Montanaro, Diego Valsesia, and Enrico Magli. Towards hyperbolic regularizers for point cloud part segmentation. In *IEEE International Conference on Acoustics, Speech and Signal Processing*, pages 1–5, 2023.
- <span id="page-10-21"></span>[59] Pierre Onghena, Leonardo Gigli, and Santiago Velasco-Forero. Rotation-invariant hierarchical segmentation on poincare ball for 3d point cloud. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 1765–1774, 2023. [2](#page-1-0)
- <span id="page-10-5"></span>[60] Yatian Pang, Wenxiao Wang, Francis EH Tay, Wei Liu, Yonghong Tian, and Li Yuan. Masked autoencoders for

- point cloud self-supervised learning. In *European Conference on Computer Vision*, pages 604–621. Springer, 2022. [1,](#page-0-0) [2,](#page-1-0) [6](#page-5-1)
- <span id="page-10-15"></span>[61] Junsung Park, Hwijeong Lee, Inha Kang, and Hyunjung Shim. No thing, nothing: Highlighting safety-critical classes for robust lidar semantic segmentation in adverse weather. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 6690–6699, 2025. [2](#page-1-0)
- <span id="page-10-9"></span>[62] Charles R Qi, Hao Su, Matthias Nießner, Angela Dai, Mengyuan Yan, and Leonidas J Guibas. Volumetric and multi-view cnns for object classification on 3d data. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 5648–5656, 2016. [2](#page-1-0)
- <span id="page-10-1"></span>[63] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 652– 660, 2017. [1,](#page-0-0) [2,](#page-1-0) [6,](#page-5-1) [4](#page-3-1)
- <span id="page-10-2"></span>[64] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. *Advances in Neural Information Processing Systems*, 30, 2017. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-10-14"></span>[65] Can Qin, Haoxuan You, Lichen Wang, C-C Jay Kuo, and Yun Fu. Pointdan: A multi-scale 3d domain adaption network for point cloud representation. *Advances in Neural Information Processing Systems*, 32, 2019. [2](#page-1-0)
- <span id="page-10-4"></span>[66] Yuzhe Qin, Binghao Huang, Zhao-Heng Yin, Hao Su, and Xiaolong Wang. Dexpoint: Generalizable point cloud reinforcement learning for sim-to-real dexterous manipulation. In *Conference on Robot Learning*, pages 594–605, 2023. [1](#page-0-0)
- <span id="page-10-3"></span>[67] Shi Qiu, Saeed Anwar, and Nick Barnes. Dense-resolution network for point cloud classification and segmentation. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, pages 3813–3822, 2021. [1](#page-0-0)
- <span id="page-10-19"></span>[68] Yiru Shen, Chen Feng, Yaoqing Yang, and Dong Tian. Mining point cloud local structures by kernel correlation and graph pooling. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 4548– 4557, 2018. [2](#page-1-0)
- <span id="page-10-11"></span>[69] Weijing Shi and Raj Rajkumar. Point-gnn: Graph neural network for 3d object detection in a point cloud. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 1711–1719, 2020. [2](#page-1-0)
- <span id="page-10-22"></span>[70] Tanuj Sur, Samrat Mukherjee, Kaizer Rahaman, Subhasis Chaudhuri, Muhammad Haris Khan, and Biplab Banerjee. Hyperbolic uncertainty-aware few-shot incremental point cloud segmentation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 11810–11821, 2025. [2](#page-1-0)
- <span id="page-10-10"></span>[71] Haotian Tang, Zhijian Liu, Shengyu Zhao, Yujun Lin, Ji Lin, Hanrui Wang, and Song Han. Searching efficient 3d architectures with sparse point-voxel convolution. In *European Conference on Computer Vision*, pages 685–702, 2020. [2](#page-1-0)
- <span id="page-10-17"></span>[72] Lyne P Tchapmi, Vineet Kosaraju, Hamid Rezatofighi, Ian Reid, and Silvio Savarese. Topnet: Structural point cloud

- decoder. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 383–392, 2019. [2](#page-1-0)
- <span id="page-11-18"></span>[73] Dorina Thanou, Philip A Chou, and Pascal Frossard. Graph-based compression of dynamic 3d point cloud sequences. *IEEE Transactions on Image Processing*, 25(4): 1765–1778, 2016. [2](#page-1-0)
- <span id="page-11-5"></span>[74] Hugues Thomas, Charles R Qi, Jean-Emmanuel Deschaud, Beatriz Marcotegui, Franc¸ois Goulette, and Leonidas J Guibas. Kpconv: Flexible and deformable convolution for point clouds. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 6411–6420, 2019. [2](#page-1-0)
- <span id="page-11-19"></span>[75] Lei Wang, Yuchun Huang, Yaolin Hou, Shenman Zhang, and Jie Shan. Graph attention convolution for point cloud semantic segmentation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 10296–10305, 2019. [2](#page-1-0)
- <span id="page-11-16"></span>[76] Lei Wang, Yuxuan Liu, Shenman Zhang, Jixing Yan, and Pengjie Tao. Structure-aware convolution for 3d point cloud classification and segmentation. *Remote Sensing*, 12(4): 634, 2020. [2](#page-1-0)
- <span id="page-11-6"></span>[77] Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E Sarma, Michael M Bronstein, and Justin M Solomon. Dynamic graph cnn for learning on point clouds. *ACM Transactions on Graphics*, 38(5):1–12, 2019. [2,](#page-1-0) [6](#page-5-1)
- <span id="page-11-20"></span>[78] Xin Wei, Ruixuan Yu, and Jian Sun. View-gcn: View-based graph convolutional network for 3d shape analysis. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 1850–1859, 2020. [2](#page-1-0)
- <span id="page-11-12"></span>[79] Xin Wei, Xiang Gu, and Jian Sun. Learning generalizable part-based feature representation for 3d point clouds. *Advances in Neural Information Processing Systems*, 35: 29305–29318, 2022. [2](#page-1-0)
- <span id="page-11-13"></span>[80] Xin Wei, Xiang Gu, and Jian Sun. Multi-scale part-based feature representation for 3d domain generalization and adaptation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 47(3):1414–1430, 2025. [2](#page-1-0)
- <span id="page-11-9"></span>[81] Bichen Wu, Alvin Wan, Xiangyu Yue, and Kurt Keutzer. Squeezeseg: Convolutional neural nets with recurrent crf for real-time road-object segmentation from 3d lidar point cloud. In *IEEE International Conference on Robotics and Automation*, pages 1887–1893, 2018. [2](#page-1-0)
- <span id="page-11-0"></span>[82] Bichen Wu, Xuanyu Zhou, Sicheng Zhao, Xiangyu Yue, and Kurt Keutzer. Squeezesegv2: Improved model structure and unsupervised domain adaptation for road-object segmentation from a lidar point cloud. In *IEEE International Conference on Robotics and Automation*, pages 4376–4382, 2019. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-11-7"></span>[83] Wenxuan Wu, Zhongang Qi, and Li Fuxin. Pointconv: Deep convolutional networks on 3d point clouds. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 9621–9630, 2019. [2](#page-1-0)
- <span id="page-11-1"></span>[84] Xiaoyang Wu, Yixing Lao, Li Jiang, Xihui Liu, and Hengshuang Zhao. Point transformer v2: Grouped vector attention and partition-based pooling. *Advances in Neural Information Processing Systems*, 35:33330–33342, 2022. [1,](#page-0-0) [2](#page-1-0)

- <span id="page-11-2"></span>[85] Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, and Hengshuang Zhao. Point transformer v3: Simpler faster stronger. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 4840–4851, 2024. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-11-14"></span>[86] Aoran Xiao, Jiaxing Huang, Weihao Xuan, Ruijie Ren, Kangcheng Liu, Dayan Guan, Abdulmotaleb El Saddik, Shijian Lu, and Eric P Xing. 3d semantic segmentation in the wild: Learning generalized models for adversecondition point clouds. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 9382–9392, 2023. [2](#page-1-0)
- <span id="page-11-15"></span>[87] Hang Xiao, Ming Cheng, and Liangwei Shi. Learning cross-domain features for domain generalization on point clouds. In *Chinese Conference on Pattern Recognition and Computer Vision*, pages 68–81, 2022. [2](#page-1-0)
- <span id="page-11-21"></span>[88] Yifan Xie, Jihua Zhu, Shiqi Li, Naiwen Hu, and Pengcheng Shi. Hecpg: Hyperbolic embedding and confident patchguided network for point cloud matching. *IEEE Transactions on Geoscience and Remote Sensing*, 62:1–12, 2024. [2](#page-1-0)
- <span id="page-11-11"></span>[89] Jiahao Xu, Xinzhu Ma, Lin Zhang, Bo Zhang, and Tao Chen. Push-and-pull: A general training framework with differential augmentor for domain generalized point cloud classification. *IEEE Transactions on Circuits and Systems for Video Technology*, 34(8):7165–7175, 2024. [2](#page-1-0)
- <span id="page-11-8"></span>[90] Mutian Xu, Runyu Ding, Hengshuang Zhao, and Xiaojuan Qi. Paconv: Position adaptive convolution with dynamic kernel assembling on point clouds. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 3173–3182, 2021. [2](#page-1-0)
- <span id="page-11-10"></span>[91] Hao Yang, Qianyu Zhou, Haijia Sun, Xiangtai Li, Fengqi Liu, Xuequan Lu, Lizhuang Ma, and Shuicheng Yan. Pointdgmamba: Domain generalization of point cloud classification via generalized state space model. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pages 9193– 9201, 2025. [2,](#page-1-0) [6,](#page-5-1) [7](#page-6-2)
- <span id="page-11-17"></span>[92] Hao Yang, Qianyu Zhou, Haijia Sun, Xiangtai Li, Xuequan Lu, Lizhuang Ma, and Shuicheng Yan. Pointdgrwkv: Generalizing rwkv-like architecture to unseen domains for point cloud classification. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pages 11595–11603, 2026. [2](#page-1-0)
- <span id="page-11-3"></span>[93] Guowen Zhang, Lue Fan, Chenhang He, Zhen Lei, Zhaoxiang Zhang, and Lei Zhang. Voxel mamba: Group-free state space models for point cloud based 3d object detection. *Advances in Neural Information Processing Systems*, 37:81489–81509, 2024. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-11-22"></span>[94] Jinlai Zhang, Lyujie Chen, Bo Ouyang, Binbin Liu, Jihong Zhu, Yujin Chen, Yanmei Meng, and Danfeng Wu. Pointcutmix: Regularization strategy for point cloud classification. *Neurocomputing*, 505:58–67, 2022. [6,](#page-5-1) [4](#page-3-1)
- <span id="page-11-4"></span>[95] Tao Zhang, Haobo Yuan, Lu Qi, Jiangning Zhang, Qianyu Zhou, Shunping Ji, Shuicheng Yan, and Xiangtai Li. Point cloud mamba: Point cloud learning via state space model. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pages 10121–10130, 2025. [1,](#page-0-0) [2](#page-1-0)

- <span id="page-12-6"></span>[96] Yingxue Zhang and Michael Rabbat. A graph-cnn for 3d point cloud classification. In *IEEE International Conference on Acoustics, Speech and Signal Processing*, pages 6279–6283, 2018. [2](#page-1-0)
- <span id="page-12-0"></span>[97] Yang Zhang, Zixiang Zhou, Philip David, Xiangyu Yue, Zerong Xi, Boqing Gong, and Hassan Foroosh. Polarnet: An improved grid representation for online lidar point clouds semantic segmentation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 9601–9610, 2020. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-12-2"></span>[98] Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip HS Torr, and Vladlen Koltun. Point transformer. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 16259–16268, 2021. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-12-5"></span>[99] Haimei Zhao, Jing Zhang, Zhuo Chen, Shanshan Zhao, and Dacheng Tao. Unimix: Towards domain adaptive and generalizable lidar semantic segmentation in adverse weather. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 14781–14791, 2024. [2](#page-1-0)
- <span id="page-12-1"></span>[100] Haoyu Zhen, Xiaowen Qiu, Peihao Chen, Jincheng Yang, Xin Yan, Yilun Du, Yining Hong, and Chuang Gan. 3d-vla: A 3d vision-language-action generative world model. In *Forty-first International Conference on Machine Learning*, 2024. [1](#page-0-0)
- <span id="page-12-3"></span>[101] Qianyu Zhou, Ke-Yue Zhang, Taiping Yao, Xuequan Lu, Ran Yi, Shouhong Ding, and Lizhuang Ma. Instance-aware domain generalization for face anti-spoofing. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 20453–20463, 2023. [2](#page-1-0)
- <span id="page-12-4"></span>[102] Qianyu Zhou, Ke-Yue Zhang, Taiping Yao, Xuequan Lu, Shouhong Ding, and Lizhuang Ma. Test-time domain generalization for face anti-spoofing. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 175–187, 2024. [2](#page-1-0)