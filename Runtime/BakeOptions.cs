using System;
using UnityEngine.Rendering;

namespace FlowFieldAI
{
    public enum BakeMode
    {
        Approach,
        Flee,
    }

    /// <summary>
    /// Configuration options for baking a flow field.
    ///
    /// Controls how many propagation steps to run, whether to allow diagonal movement,
    /// and how the workload is distributed across frames.
    /// </summary>
    [Serializable]
    public struct BakeOptions
    {
        /// <summary>
        /// Total number of propagation steps (iterations) to run.
        /// A higher number increases the max reachable distance.
        /// </summary>
        public int Iterations;

        /// <summary>
        /// Number of iterations to run per frame.
        /// Set to 0 to run the entire bake in a single frame.
        /// </summary>
        public int IterationsPerFrame;

        /// <summary>
        /// Enables diagonal movement in addition to cardinal directions.
        /// </summary>
        public bool DiagonalMovement;

        /// <summary>
        /// GPU queue used for compute dispatch. Currently not effective for async compute.
        /// </summary>
        public ComputeQueueType ComputeQueueType;

        public BakeMode Mode;

        public float FleeFactor;

        /// <summary>
        /// Returns true if the bake will be performed incrementally across multiple frames.
        /// </summary>
        public bool IsIncrementalBake => IterationsPerFrame > 0 && IterationsPerFrame < Iterations;

        /// <summary>
        /// Recommended default settings:
        /// 100 iterations, full bake in a single frame, no diagonals.
        /// </summary>
        public static BakeOptions Default => new()
        {
            Iterations = 100,
            IterationsPerFrame = 0,
            DiagonalMovement = false,
            ComputeQueueType = ComputeQueueType.Background,
            Mode = BakeMode.Approach,
            FleeFactor = -1.2f,
        };
    }
}
