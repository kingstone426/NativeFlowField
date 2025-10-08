using System;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;
using System.Diagnostics;
using Unity.Entities;
using Debug = UnityEngine.Debug;
using System.Linq;

namespace FlowFieldAI
{
    /// <summary>
    /// GPU-accelerated flow field generator for grid-based navigation.
    ///
    /// Computes movement directions from any point on the grid to one or more targets,
    /// using a distance field propagated on the GPU.
    ///
    /// Designed for large agent counts, dynamic obstacles, and real-time updates.
    /// Results are written to a NativeArray and optionally a RenderTexture heatmap.
    ///
    /// Can be attached to entities as a managed component for usage in Unity DOTS workflows.
    /// </summary>
    public class NativeFlowField : IComponentData, IDisposable
    {
        // ─────────────────────────────────────────────────────────────
        // Public Properties
        // ─────────────────────────────────────────────────────────────

        /// <summary>
        /// The final flow field result. Each index maps to the next cell to move to in order to approach the nearest target.
        /// </summary>
        public NativeArray<int> NextIndices { get; private set; }

        /// <summary>
        /// A RenderTexture visualization of the latest distance propagation heat map. Optional; generated only if requested.
        /// </summary>
        public RenderTexture HeatMap => heatMapFrontBuffer;

        /// <summary>
        /// The width of the flow field, in cells.
        /// </summary>
        public readonly int Width;

        /// <summary>
        /// The height of the flow field, in cells.
        /// </summary>
        public readonly int Height;

        /// <summary>
        /// Total number of cells in the flow field. Equals Width × Height.
        /// </summary>
        public int Length => Width * Height;

        /// <summary>
        /// Number of frames elapsed between a Bake call and final data availability (readback complete).
        /// </summary>
        public int BakeFrameLatency { get; private set; }

        /// <summary>
        /// Time elapsed in seconds between a Bake call and final data availability (readback complete).
        /// </summary>
        public float BakeTimeLatency { get; private set; }

        /// <summary>
        /// Time elapsed in seconds on the main thread during the last Bake call.
        /// </summary>
        public float BakeDispatchTime { get; private set; }

        /// <summary>
        /// Number of internal GPU buffers currently in use for overlapping async compute dispatches.
        /// </summary>
        public int BuffersActive => bufferPool.RentedOutCount;

        /// <summary>
        /// Total number of GPU buffers currently allocated.
        /// </summary>
        public int BuffersAllocated => bufferPool.Count;

        /// <summary>
        /// Maximum allowed number of internal GPU buffers in the pool.
        /// </summary>
        public int BuffersCapacity => bufferPool.Capacity;

        // ─────────────────────────────────────────────────────────────
        // Constants
        // ─────────────────────────────────────────────────────────────

        /// <summary>
        /// Sentinel value used to mark impassable cells (obstacles) in the input field.
        /// </summary>
        public const float ObstacleCell = float.MaxValue;

        /// <summary>
        /// Sentinel value used to mark walkable, non-target cells in the input field.
        /// </summary>
        public const float FreeCell = float.MinValue;

        private static class ShaderProperties
        {
            public static readonly int Width = Shader.PropertyToID("Width");
            public static readonly int Height = Shader.PropertyToID("Height");
            public static readonly int DiagonalMovement = Shader.PropertyToID("DiagonalMovement");
            public static readonly int UseTravelCosts = Shader.PropertyToID("UseTravelCosts");
            public static readonly int TravelCosts = Shader.PropertyToID("TravelCosts");
            public static readonly int InputCosts = Shader.PropertyToID("InputCosts");
            public static readonly int OutputCosts = Shader.PropertyToID("OutputCosts");
            public static readonly int OutputHeatMap = Shader.PropertyToID("OutputHeatMap");
            public static readonly int OutputFlowField = Shader.PropertyToID("OutputFlowField");
        }

        // ─────────────────────────────────────────────────────────────
        // Private Fields
        // ─────────────────────────────────────────────────────────────
        private ComputeBuffer travelCostsBuffer;
        private readonly bool useTravelCosts;

        private readonly ComputeShader integrationComputeShader;
        private readonly int integrationComputeShaderKernel;
        private ComputeBuffer integrationFrontBuffer;
        private ComputeBuffer integrationBackBuffer;

        private readonly ComputeShader generateFlowFieldComputeShader;
        private readonly int generateFlowFieldComputeShaderKernel;
        private readonly ComputeBuffer flowFieldBuffer;

        private readonly ComputeShader generateHeatMapComputeShader;
        private readonly int generateHeatMapComputeShaderKernel;
        private readonly bool generateHeatMap;
        private readonly RenderTexture heatMapBackBuffer;
        private readonly RenderTexture heatMapFrontBuffer;

        private readonly NativeArrayPool bufferPool;
        private int frontBufferPoolIndex;

        private BakeContext bakeContext;
        private readonly CommandBuffer commandBuffer = new();
        private readonly Stopwatch bakeTimer = Stopwatch.StartNew();

        // ─────────────────────────────────────────────────────────────
        // Constructor
        // ─────────────────────────────────────────────────────────────

        /// <summary>
        /// Default constructor required for IComponentData serialization.
        /// </summary>
        /// <exception cref="NotImplementedException"></exception>
        public NativeFlowField() : this(1, 1) => throw new NotImplementedException();

        /// <summary>
        /// Creates a new flow field instance with the specified resolution and optional GPU heat map output.
        /// </summary>
        /// <param name="width">Field width in cells.</param>
        /// <param name="height">Field height in cells.</param>
        /// <param name="generateHeatMap">If true, allocates an additional render texture to visualize distance propagation.</param>
        /// <param name="minBuffers">Minimum number of buffers to keep pooled.</param>
        /// <param name="maxBuffers">Maximum number of buffers allowed for async dispatching.</param>
        public NativeFlowField(int width, int height, bool generateHeatMap=false, bool useTravelCosts=false, int minBuffers=2, int maxBuffers=5)
        {
            if (minBuffers < 2)
            {
                throw new ArgumentException("minBuffers must be at least 2");
            }

            if (maxBuffers < minBuffers)
            {
                throw new ArgumentException("maxBuffers must be at least minBuffers");
            }

            Width = width;
            Height = height;

            this.generateHeatMap = generateHeatMap;

            travelCostsBuffer = new ComputeBuffer(Length, sizeof(float));
            this.useTravelCosts = useTravelCosts;

            integrationComputeShader = Resources.Load<ComputeShader>("GenerateIntegrationField");
            integrationComputeShaderKernel = integrationComputeShader.FindKernel("GenerateIntegrationField");

            integrationFrontBuffer = new ComputeBuffer(Length, sizeof(float));
            integrationBackBuffer = new ComputeBuffer(Length, sizeof(float));

            generateFlowFieldComputeShader = Resources.Load<ComputeShader>("GenerateFlowField");
            generateFlowFieldComputeShaderKernel = generateFlowFieldComputeShader.FindKernel("GenerateFlowField");
            flowFieldBuffer = new ComputeBuffer(Length, sizeof(int));

            if (generateHeatMap)
            {
                generateHeatMapComputeShader = Resources.Load<ComputeShader>("GenerateHeatMap");
                generateHeatMapComputeShaderKernel = generateHeatMapComputeShader.FindKernel("GenerateHeatMap");

                heatMapBackBuffer = new RenderTexture(width, height, 0, RenderTextureFormat.RFloat)
                {
                    enableRandomWrite = true,
                    filterMode = FilterMode.Point,
                };
                heatMapFrontBuffer = new RenderTexture(width, height, 0, RenderTextureFormat.RFloat)
                {
                    enableRandomWrite = true,
                    filterMode = FilterMode.Point,
                };
            }

            bufferPool = new NativeArrayPool(minBuffers, maxBuffers, Length);
        }

        // ─────────────────────────────────────────────────────────────
        // Public Methods
        // ─────────────────────────────────────────────────────────────

        /// <summary>
        /// Releases all GPU and native memory resources associated with this flow field instance.
        /// </summary>
        public void Dispose()
        {
            travelCostsBuffer.Dispose();
            integrationFrontBuffer.Dispose();
            integrationBackBuffer.Dispose();
            flowFieldBuffer.Dispose();
            bufferPool.Dispose();
            heatMapBackBuffer?.Release();
            heatMapFrontBuffer?.Release();
        }

        /// <summary>
        /// Starts a new asynchronous bake using the default options.
        /// The returned request can be used to track GPU readback completion.
        /// </summary>
        /// <param name="inputField">
        /// A native array representing the input field. Use <see cref="ObstacleCell"/> and <see cref="FreeCell"/> for special cells,
        /// and numerical values for target weights.
        /// </param>
        /// <returns>An async GPU readback request that completes when the field is ready.</returns>
        public AsyncGPUReadbackRequest Bake(NativeArray<float> inputField) => Bake(inputField, BakeOptions.Default);

        /// <summary>
        /// Starts a new asynchronous bake using the specified bake options and 0-initialized travel costs.
        /// The returned request can be used to track GPU readback completion.
        /// </summary>
        /// <param name="inputField">
        /// A native array representing the input field. Use <see cref="ObstacleCell"/> and <see cref="FreeCell"/> for special cells,
        /// and numerical values for target weights.
        /// </param>
        /// <param name="bakeOptions">Configuration options for this bake operation.</param>
        /// <returns>An async GPU readback request that completes when the field is ready.</returns>
        public AsyncGPUReadbackRequest Bake(NativeArray<float> inputField, BakeOptions bakeOptions) => Bake(inputField, new NativeArray<float>(), bakeOptions);

        /// <summary>
        /// Starts a new asynchronous bake using the specified bake options.
        /// The returned request can be used to track GPU readback completion.
        /// </summary>
        /// <param name="inputField">
        /// A native array representing the input field. Use <see cref="ObstacleCell"/> and <see cref="FreeCell"/> for special cells,
        /// and numerical values for target weights.
        /// </param>
        /// <param name="travelCosts">
        /// A native array representing the travel costs of each cell in the input field.
        /// These costs are added to the minimum weight calculation of each target cell at the same index.
        /// </param>
        /// <param name="bakeOptions">Configuration options for this bake operation.</param>
        /// <returns>An async GPU readback request that completes when the field is ready.</returns>
        public AsyncGPUReadbackRequest Bake(NativeArray<float> inputField, NativeArray<float> travelCosts, BakeOptions bakeOptions)
        {
            bakeTimer.Restart();

            ValidateMatrixDimensions(inputField);
            if (useTravelCosts) ValidateMatrixDimensions(travelCosts);

            if (bufferPool.IsExhausted)
            {
                BakeDispatchTime = (float)bakeTimer.Elapsed.TotalSeconds;
                return default;
            }

            // Initialize buffers
            var iterationsThisFrame = InitializeBake(inputField, travelCosts, bakeOptions);

            // Generate integration field
            PerformIntegration(iterationsThisFrame);

            // Do not finalize flow field and heat map until incremental dispatch is complete.
            if (bakeContext.Options.IsIncrementalBake && bakeContext.CurrentIteration < bakeContext.Options.Iterations)
            {
                BakeDispatchTime = (float)bakeTimer.Elapsed.TotalSeconds;
                return default;
            }

            // Generate flow field
            GenerateFlowField();

            // Generate heat map
            GenerateHeatMap();

            // Prepare async GPU readback
            var poolIndex = bufferPool.Rent();
            var buffer = bufferPool[poolIndex];
            var dispatchFrame = Time.frameCount;
            var dispatchTime = Time.realtimeSinceStartup;

            BakeDispatchTime = (float)bakeTimer.Elapsed.TotalSeconds;

            // Dispatch async readback request
            return AsyncGPUReadback.RequestIntoNativeArray(
                ref buffer,
                flowFieldBuffer,
                Length * sizeof(uint),
                0,
                req => OnReadbackComplete(poolIndex, req, dispatchFrame, dispatchTime));
        }

        // ─────────────────────────────────────────────────────────────
        // Private Methods
        // ─────────────────────────────────────────────────────────────
        private int InitializeBake(NativeArray<float> inputField, NativeArray<float> travelCostsField, BakeOptions bakeOptions)
        {
            // Initialize graphics command buffer
            commandBuffer.Clear();
            commandBuffer.name = "NativeFlowField CommandBuffer";
            commandBuffer.SetExecutionFlags(CommandBufferExecutionFlags.AsyncCompute);

            // Do not reset bake context if incremental bake is in progress.
            var initializeNewBake = !bakeOptions.IsIncrementalBake || bakeContext == null;
            if (initializeNewBake)
            {
                bakeContext = new BakeContext
                {
                    Options = bakeOptions,
                    CurrentIteration = 0,
                };
            }

            // Calculate iteration count
            var iterationsRemaining = bakeOptions.Iterations - bakeContext.CurrentIteration;
            var iterationsThisFrame = bakeOptions.IsIncrementalBake
                ? Mathf.Min(iterationsRemaining, bakeOptions.IterationsPerFrame)
                : Mathf.Min(iterationsRemaining, bakeOptions.Iterations);

            // Initialize compute buffer with input field
            if (bakeContext.CurrentIteration == 0 && initializeNewBake)
            {
                commandBuffer.SetBufferData(integrationFrontBuffer, inputField);
                if (useTravelCosts)
                    commandBuffer.SetBufferData(travelCostsBuffer, travelCostsField);
            }

            return iterationsThisFrame;
        }

        private void PerformIntegration(int iterationsThisFrame)
        {
            commandBuffer.SetComputeIntParam(integrationComputeShader, ShaderProperties.Width, Width);
            commandBuffer.SetComputeIntParam(integrationComputeShader, ShaderProperties.Height, Height);
            commandBuffer.SetComputeIntParam(integrationComputeShader, ShaderProperties.DiagonalMovement, bakeContext.Options.DiagonalMovement ? 1 : 0);
            commandBuffer.SetComputeIntParam(integrationComputeShader, ShaderProperties.UseTravelCosts, useTravelCosts ? 1 : 0);

            // Assign travel cost buffer
            commandBuffer.SetComputeBufferParam(integrationComputeShader, integrationComputeShaderKernel, ShaderProperties.TravelCosts, travelCostsBuffer);

            for (var i = 0; i < iterationsThisFrame; i++)
            {
                // Assign compute buffers
                commandBuffer.SetComputeBufferParam(integrationComputeShader, integrationComputeShaderKernel, ShaderProperties.InputCosts, integrationFrontBuffer);
                commandBuffer.SetComputeBufferParam(integrationComputeShader, integrationComputeShaderKernel, ShaderProperties.OutputCosts, integrationBackBuffer);

                // Dispatch compute shader
                var threadGroupsX = Mathf.CeilToInt(Width / 8f);
                var threadGroupsY = Mathf.CeilToInt(Height / 8f);
                commandBuffer.DispatchCompute(integrationComputeShader, integrationComputeShaderKernel, threadGroupsX, threadGroupsY, 1);

                // Flip compute buffers
                (integrationFrontBuffer, integrationBackBuffer) = (integrationBackBuffer, integrationFrontBuffer);

                // Increment step counter
                bakeContext.CurrentIteration++;
            }

            Graphics.ExecuteCommandBufferAsync(commandBuffer, bakeContext.Options.ComputeQueueType);
        }

        private void GenerateFlowField()
        {
            var threadGroupsX = Mathf.CeilToInt(Width / 8f);
            var threadGroupsY = Mathf.CeilToInt(Height / 8f);

            commandBuffer.Clear();
            commandBuffer.name = "NativeFlowField CommandBuffer - GenerateFlowField";
            commandBuffer.SetExecutionFlags(CommandBufferExecutionFlags.AsyncCompute);

            commandBuffer.SetComputeIntParam(generateFlowFieldComputeShader, ShaderProperties.Width, Width);
            commandBuffer.SetComputeIntParam(generateFlowFieldComputeShader, ShaderProperties.Height, Height);
            commandBuffer.SetComputeIntParam(generateFlowFieldComputeShader, ShaderProperties.DiagonalMovement, bakeContext.Options.DiagonalMovement ? 1 : 0);
            commandBuffer.SetComputeIntParam(generateFlowFieldComputeShader, ShaderProperties.UseTravelCosts, useTravelCosts ? 1 : 0);
            commandBuffer.SetComputeBufferParam(generateFlowFieldComputeShader, generateFlowFieldComputeShaderKernel, ShaderProperties.InputCosts, integrationFrontBuffer);
            commandBuffer.SetComputeBufferParam(generateFlowFieldComputeShader, generateFlowFieldComputeShaderKernel, ShaderProperties.OutputFlowField, flowFieldBuffer);
            commandBuffer.SetComputeBufferParam(generateFlowFieldComputeShader, generateFlowFieldComputeShaderKernel, ShaderProperties.TravelCosts, travelCostsBuffer);
            commandBuffer.DispatchCompute(generateFlowFieldComputeShader, generateFlowFieldComputeShaderKernel, threadGroupsX, threadGroupsY, 1);

            Graphics.ExecuteCommandBufferAsync(commandBuffer, bakeContext.Options.ComputeQueueType);
        }

        private void GenerateHeatMap()
        {
            if (!generateHeatMap)
            {
                return;
            }

            var threadGroupsX = Mathf.CeilToInt(Width / 8f);
            var threadGroupsY = Mathf.CeilToInt(Height / 8f);

            commandBuffer.Clear();
            commandBuffer.name = "NativeFlowField CommandBuffer - GenerateHeatMap";
            commandBuffer.SetExecutionFlags(CommandBufferExecutionFlags.AsyncCompute);

            commandBuffer.SetComputeIntParam(generateHeatMapComputeShader, ShaderProperties.Width, Width);
            commandBuffer.SetComputeIntParam(generateHeatMapComputeShader, ShaderProperties.Height, Height);
            commandBuffer.SetComputeTextureParam(generateHeatMapComputeShader, generateHeatMapComputeShaderKernel, ShaderProperties.OutputHeatMap, heatMapBackBuffer);
            commandBuffer.SetComputeBufferParam(generateHeatMapComputeShader, generateHeatMapComputeShaderKernel, ShaderProperties.InputCosts, integrationFrontBuffer);
            commandBuffer.DispatchCompute(generateHeatMapComputeShader, generateHeatMapComputeShaderKernel, threadGroupsX, threadGroupsY, 1);

            Graphics.ExecuteCommandBufferAsync(commandBuffer, bakeContext.Options.ComputeQueueType);
        }

        private void OnReadbackComplete(int poolIndex, AsyncGPUReadbackRequest req, int dispatchFrame, float dispatchTime)
        {
            if (req.hasError)
            {
                Debug.LogError("GPU readback error!");
                return;
            }

            // Update latency
            BakeFrameLatency = Time.frameCount - dispatchFrame;
            BakeTimeLatency = Time.realtimeSinceStartup - dispatchTime;

            // Return front buffer to pool
            if (NextIndices.IsCreated)
            {
                bufferPool.Return(frontBufferPoolIndex);
            }

            // Present front buffer
            NextIndices = bufferPool[poolIndex];
            frontBufferPoolIndex = poolIndex;

            // Present heat map
            if (generateHeatMap && heatMapBackBuffer && heatMapFrontBuffer)
            {
                Graphics.CopyTexture(heatMapBackBuffer, heatMapFrontBuffer);
            }

            // Clear bake context
            bakeContext = null;
        }

        [Conditional("ENABLE_UNITY_COLLECTIONS_CHECKS")]
        private void ValidateMatrixDimensions(NativeArray<float> matrix)
        {
            if (matrix.Length != Length)
            {
                throw new ArgumentException($"NativeArray length ({matrix.Length}) mismatching " +
                                            $"NativeDijkstraMap dimensions ({Width} x {Height}) = {Length}.");
            }
        }
    }
}

