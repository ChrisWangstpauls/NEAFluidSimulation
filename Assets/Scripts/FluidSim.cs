using UnityEngine;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Burst;
using System;
using System.Text.RegularExpressions;

public class FluidSimulation : MonoBehaviour
{
	[Header("Runtime Logging Settings")]
	[Tooltip("Enable real-time metrics logging to SQLite")]
	[SerializeField] private bool _enableRuntimeLogging = true;

	[Tooltip("How often to log data (in simulation steps)")]
	[SerializeField] private int _loggingInterval = 10;

	[Header("Simulation Parameters")]
	public bool paused = false;
	[Range(32, 512)]
	public int size = 128;
	[Tooltip("Physical size of the simulation area")]
	public float physicalSize = 1.0f;
	[Range(0.1f, 10f)]
	public float resolutionMultiplier = 1.0f;
	public float diffusion = 0.0001f;
	public float viscosity = 0.0001f;
	public float timeStep = 0.1f;
	public bool autoAdjustParameters = true;
	public bool applyTurbulentNoise = false;
	public enum ColorMode { SingleColor, Gradient, DensityBased, PressureBased, Streamlines }

	[Header("Customizable Source")]
	public bool enableCustomSource = false;
	[Range(1f, 500f)]
	public float sourceStrength = 100f;
	public bool sourceEmitsVelocity = false;
	[Range(0f, 360f)]
	public float sourceDirection = 0f;
	[Range(1f, 50f)]
	public float sourceVelocity = 10f;
	[Range(0.1f, 10f)]
	public float sourceRadius = 1f;
	[Range(0.1f, 5f)]
	public float sourcePulseRate = 1f;
	public bool sourcePulsing = false;
	[Range(0f, 1f)]
	public float sourcePositionX = 0.5f; // Normalized position (0-1)
	[Range(0f, 1f)]
	public float sourcePositionY = 0.5f; // Normalized position (0-1)
	public bool moveSourceWithMouse = false;
	public KeyCode sourcePositionKey = KeyCode.LeftShift; // Hold key to position source with mouse
	public bool visualizeSourcePosition = true;
	public Color sourcePositionColor = Color.yellow;

	[Header("Single Colour Visualization")]
	public Color fluidcolour = Color.white;
	[Range(0f, 1f)]
	public float colourIntensity = 1f;
	public Gradient colourGradient;
	public bool useLerp = false;
	public Color startColor = Color.white;
	public Color endColor = Color.white;

	[Header("Pressure Visualization")]
	public Color lowPressureColor = Color.blue;
	public Color neutralPressureColor = Color.white;
	public Color highPressureColor = Color.red;
	[Range(0f, 1f)]
	public float lowPressureThreshold = -50f;
	[Range(-0f, 1f)]
	public float highPressureThreshold = 50f;

	[Header("Density Settings")]
	public ColorMode colorMode = ColorMode.SingleColor;
	public Color lowDensityColor = Color.blue;
	public Color mediumDensityColor = Color.green;
	public Color highDensityColor = Color.red;
	[Range(0f, 500f)]
	public float mediumDensityThreshold = 50f;
	[Range(0f, 1000f)]
	public float highDensityThreshold = 200f;

	[Header("Streamline Visualization")]
	public bool showStreamlines = false;
	[Range(1f, 5f)]
	public int streamlineDensity = 4; // Controls how many cells are skipped
	[Range(1f, 10f)]
	public float streamlineScale = 1.0f; // Scales length of the lines
	public Color streamlineColor = Color.white;
	[Range(0.1f, 3f)]
	public float streamlineThickness = 1.0f;
	private Texture2D streamlineTexture;

	[Header("Obstacle Settings")]
	public bool enableObstacle = true;
	public enum ObstacleShape { Circle, Rectangle, Airfoil }
	public ObstacleShape obstacleShape = ObstacleShape.Circle;
	[Range(0f, 1f)]
	public float obstaclePositionX = 0.5f;
	[Range(0f, 1f)]
	public float obstaclePositionY = 0.5f;
	[Range(0.01f, 0.5f)]
	public float obstacleRadius = 0.1f;
	[Range(0.01f, 0.5f)]
	public float obstacleWidth = 0.2f;
	[Range(0.01f, 0.5f)]
	public float obstacleHeight = 0.2f;
	public Color obstacleColor = Color.gray;

	private float[] density;
	private float[] velocityX;
	private float[] velocityY;
	private float[] velocityX0;
	private float[] velocityY0;
	private float[] pressure;

	private int currentSize;
	private float cellSize;
	private float dtScale;
	private float elapsedTime = 0f;

	private Material fluidMaterial;
	private Texture2D fluidTexture;
	private Camera mainCamera;
	private Vector3[] quadCorners = new Vector3[4];

	private int previousSize;
	private float previousResolutionMultiplier;

	private bool[] obstacles;

	private NativeArray<float4> streamlineData;
	private bool streamlineBuffersInitialized = false;

	private NativeArray<float> jobBuffer1;
	private NativeArray<float> jobBuffer2;
	private NativeArray<bool> jobObstacles;
	private bool jobBuffersInitialized = false;
	private int currentStep = 0;
	private int _currentRunID = -1;

	private float _smoothedFPS;
	private const float SMOOTH_FACTOR = 0.9f;
	private Vector2 _prevMouseGridPos;
	private bool _isFirstDragFrame = true;

	public void SetPaused(bool Paused)
	{
		paused = Paused;
		Debug.Log($"Simulation paused: {paused}");
	}
	void OnValidate()
	{
		// Update resolution when parameters change in editor
		if (previousSize != size || previousResolutionMultiplier != resolutionMultiplier)
		{
			// Only reinitialize if already started
			if (fluidTexture != null)
			{
				ResetSimulation();
			}

			previousSize = size;
			previousResolutionMultiplier = resolutionMultiplier;
		}

		// Update visualization when parameters change in editor
		if (fluidTexture != null)
		{
			UpdateVisualization();
		}

		// Update obstacles when parameters change in editor
		if (obstacles != null && obstacles.Length > 0)
		{
			SetupObstacles();
		}
	}

	void Start()
	{
		ResetSimulation();
		SetupObstacles();

		// Initialize default gradient if none is set
		if (colourGradient.Equals(new Gradient()))
		{
			GradientColorKey[] colourKeys = new GradientColorKey[2];
			colourKeys[0].color = Color.blue;
			colourKeys[0].time = 0.0f;
			colourKeys[1].color = Color.red;
			colourKeys[1].time = 1.0f;

			GradientAlphaKey[] alphaKeys = new GradientAlphaKey[2];
			alphaKeys[0].alpha = 1.0f;
			alphaKeys[0].time = 0.0f;
			alphaKeys[1].alpha = 1.0f;
			alphaKeys[1].time = 1.0f;

			colourGradient.SetKeys(colourKeys, alphaKeys);
		}

		_currentRunID = SQL.SaveSimRunParams(
		size, diffusion, viscosity, timeStep,
		enableCustomSource, sourceStrength, sourcePositionX, sourcePositionY,
		enableObstacle, obstacleShape.ToString(), obstaclePositionX, obstaclePositionY,
		obstacleRadius, obstacleWidth, obstacleHeight
		);
	}

	void ResetSimulation()
	{
		// Calculate actual simulation size based on resolution multiplier
		currentSize = Mathf.RoundToInt(size * resolutionMultiplier);

		// Calculate cell size for physical accuracy
		cellSize = physicalSize / currentSize;

		// Scale time step inversely with resolution to maintain stability	
		dtScale = autoAdjustParameters ? 128f / currentSize : 1f;

		// Initialize arrays
		int totalSize = currentSize * currentSize;
		density = new float[totalSize];
		pressure = new float[totalSize];
		velocityX = new float[totalSize];
		velocityY = new float[totalSize];
		velocityX0 = new float[totalSize];
		velocityY0 = new float[totalSize];
		obstacles = new bool[totalSize];

		// Reset job buffers when simulation size changes
		ResetJobBuffers();

		// Create or recreate visualization texture
		if (fluidTexture != null)
		{
			Destroy(fluidTexture);
		}
		fluidTexture = new Texture2D(currentSize, currentSize);
		fluidTexture.filterMode = FilterMode.Bilinear;

		// Create or update material for rendering
		if (fluidMaterial == null)
		{
			fluidMaterial = new Material(Shader.Find("Unlit/Texture"));
		}
		fluidMaterial.mainTexture = fluidTexture;

		// Create quad for visualization if it doesn't exist
		GameObject quad = GameObject.Find("FluidSimulationQuad");
		if (quad == null)
		{
			quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
			quad.name = "FluidSimulationQuad";
			quad.GetComponent<Renderer>().material = fluidMaterial;
		}
		else
		{
			quad.GetComponent<Renderer>().material = fluidMaterial;
		}

		mainCamera = Camera.main;

		// Store quad mesh filter for corner calculations
		MeshFilter meshFilter = quad.GetComponent<MeshFilter>();
		List<Vector3> cornersList = new List<Vector3>();
		meshFilter.mesh.GetVertices(cornersList);
		quadCorners = cornersList.ToArray();

		// Transform corners to world space
		for (int i = 0; i < 4; i++)
		{
			quadCorners[i] = quad.transform.TransformPoint(quadCorners[i]);
		}

#if UNITY_EDITOR
		Debug.Log($"Fluid simulation reset with resolution: {currentSize}x{currentSize}, cell size: {cellSize}");
#endif
		string logMessage = $"Fluid simulation reset with resolution: {currentSize}x{currentSize}, cell size: {cellSize}";
		Match match = Regex.Match(logMessage, @"resolution:\s*(\d+)x(\d+),\s*cell size:\s*([\d.]+)");
		if (match.Success)
		{
			int loggedSizeX = int.Parse(match.Groups[1].Value);
			int loggedSizeY = int.Parse(match.Groups[2].Value);
			// Compare with currentSize/cellSize to detect mismatches
		}

		// Initialize streamline texture if needed
		if (streamlineTexture != null)
		{
			Destroy(streamlineTexture);
		}
		streamlineTexture = new Texture2D(currentSize, currentSize, TextureFormat.RGBA32, false);
		streamlineTexture.filterMode = FilterMode.Point;

		SetupObstacles();
	}

	void SetupObstacles()
	{
		// Clear all obstacles
		Array.Clear(obstacles, 0, obstacles.Length);

		if (!enableObstacle) return;

		int startX = Mathf.RoundToInt(obstaclePositionX * currentSize);
		int startY = Mathf.RoundToInt(obstaclePositionY * currentSize);

		// Recursive flood fill for different obstacle shapes
		switch (obstacleShape)
		{
			case ObstacleShape.Circle:
				RecursiveFloodFill(startX, startY, obstacleRadius * currentSize, ObstacleShape.Circle);
				break;

			case ObstacleShape.Rectangle:
				RecursiveFloodFill(startX, startY, obstacleWidth * currentSize, ObstacleShape.Rectangle);
				break;

			case ObstacleShape.Airfoil:
				RecursiveFloodFill(startX, startY, obstacleWidth * currentSize, ObstacleShape.Airfoil);
				break;
		}
	}

	void RecursiveFloodFill(int x, int y, float size, ObstacleShape shape)
	{
		// Boundary checks
		if (x < 0 || x >= currentSize || y < 0 || y >= currentSize)
			return;

		int index = GridUtils.IX(x, y, currentSize);

		// Stop if already marked or out of bounds
		if (obstacles[index]) return;

		// Determine if the cell should be part of the obstacle
		if (!IsInsideShape(x, y, size, shape)) return;

		// Mark the current cell as an obstacle
		obstacles[index] = true;

		// Recursively call flood fill in four directions
		RecursiveFloodFill(x + 1, y, size, shape);
		RecursiveFloodFill(x - 1, y, size, shape);
		RecursiveFloodFill(x, y + 1, size, shape);
		RecursiveFloodFill(x, y - 1, size, shape);
	}

	bool IsInsideShape(int x, int y, float size, ObstacleShape shape)
	{
		float centerX = obstaclePositionX * currentSize;
		float centerY = obstaclePositionY * currentSize;

		switch (shape)
		{
			case ObstacleShape.Circle:
				return (x - centerX) * (x - centerX) + (y - centerY) * (y - centerY) < size * size;

			case ObstacleShape.Rectangle:
				float halfWidth = obstacleWidth * currentSize * 0.5f;
				float halfHeight = obstacleHeight * currentSize * 0.5f;
				return x > (centerX - halfWidth) && x < (centerX + halfWidth) &&
					   y > (centerY - halfHeight) && y < (centerY + halfHeight);

			case ObstacleShape.Airfoil:
				// Approximate NACA 0015 airfoil shape
				float chord = 2 * obstacleWidth * currentSize;
				float thickness = 0.15f;
				float normX = (x - centerX + chord / 2) / chord;
				float normY = (y - centerY) / chord;

				if (normX < 0 || normX > 1 || Math.Abs(normY) > thickness)
					return false;

				float halfThickness = 5 * thickness * (0.2969f * Mathf.Sqrt(normX) - 0.1260f * normX -
								   0.3516f * normX * normX + 0.2843f * normX * normX * normX -
								   0.1015f * normX * normX * normX * normX);

				return Math.Abs(normY) <= halfThickness;

			default:
				return false;
		}
	}

	void Update()
	{
		if (paused) return;

		elapsedTime += Time.deltaTime;

		// Handle source positioning with mouse if enabled
		if (moveSourceWithMouse && Input.GetKey(sourcePositionKey))
		{
			Vector2 mousePos = GetMousePositionInGrid();
			sourcePositionX = Mathf.Clamp01(mousePos.x / currentSize);
			sourcePositionY = Mathf.Clamp01(mousePos.y / currentSize);
		}

		// Process custom source if enabled
		if (enableCustomSource)
		{
			UpdateCustomSource();
		}

		// Get current mouse position in grid coordinates
		Vector2 currentMousePos = GetMousePositionInGrid();

		// Process mouse drag input
		if (Input.GetMouseButton(0) && !(moveSourceWithMouse && Input.GetKey(sourcePositionKey)))
		{
			if (!_isFirstDragFrame)
			{
				// Calculate velocity based on position delta
				Vector2 mouseDelta = currentMousePos - _prevMouseGridPos;
				Vector2 mouseVelocity = mouseDelta / Time.deltaTime;

				// Calculate force direction and magnitude
				float forceMagnitude = mouseDelta.magnitude * resolutionMultiplier;
				Vector2 forceDirection = mouseDelta.normalized;

				// Apply force with non-linear response curve
				float scaledForce = Mathf.Pow(forceMagnitude, 1.5f) * 0.8f;
				AddForceToArea(
					currentMousePos,
					forceDirection * scaledForce,
					Mathf.Clamp(forceMagnitude * 0.5f, 2f, 10f)
				);
			}
			_isFirstDragFrame = false;
			_prevMouseGridPos = currentMousePos;
		}
		else
		{
			_isFirstDragFrame = true;
		}

		Simulate();
		UpdateVisualization();

		// Added DrawStreamlines call if needed separately
		if (showStreamlines && colorMode != ColorMode.Streamlines)
		{
			CombineTextures();
		}
	}

	private void AddForceToArea(Vector2 center, Vector2 force, float radius)
	{
		int minX = Mathf.Clamp((int)(center.x - radius), 0, currentSize - 1);
		int maxX = Mathf.Clamp((int)(center.x + radius), 0, currentSize - 1);
		int minY = Mathf.Clamp((int)(center.y - radius), 0, currentSize - 1);
		int maxY = Mathf.Clamp((int)(center.y + radius), 0, currentSize - 1);

		for (int x = minX; x <= maxX; x++)
		{
			for (int y = minY; y <= maxY; y++)
			{
				Vector2 cellPos = new Vector2(x, y);
				float distance = Vector2.Distance(cellPos, center);

				if (distance <= radius)
				{
					float falloff = 1 - (distance / radius);
					int index = GridUtils.IX(x, y, currentSize);

					// Apply force with smooth falloff
					velocityX[index] += force.x * falloff;
					velocityY[index] += force.y * falloff;

					// Add density at the center
					if (distance < radius * 0.3f)
					{
						density[index] += sourceStrength * falloff;
					}
				}
			}
		}
	}

	void UpdateCustomSource()
	{
		// Convert normalized position (0-1) to grid coordinates
		float sourceX = sourcePositionX * currentSize;
		float sourceY = sourcePositionY * currentSize;

		// Calculate effective strength for pulsing if enabled
		float pulseScale = sourcePulsing
			? Mathf.Abs(Mathf.Sin(elapsedTime * sourcePulseRate * Mathf.PI))
			: 1f;
		float effectiveStrength = sourceStrength * pulseScale;

		// Adjust strength based on resolution
		effectiveStrength *= resolutionMultiplier;

		// Apply density to source with radius
		float radiusInCells = sourceRadius * resolutionMultiplier;

		for (int i = Mathf.Max(0, Mathf.FloorToInt(sourceX - radiusInCells));
			 i <= Mathf.Min(currentSize - 1, Mathf.CeilToInt(sourceX + radiusInCells)); i++)
		{
			for (int j = Mathf.Max(0, Mathf.FloorToInt(sourceY - radiusInCells));
				 j <= Mathf.Min(currentSize - 1, Mathf.CeilToInt(sourceY + radiusInCells)); j++)
			{
				// Calculate distance from source point
				float distSq = (i - sourceX) * (i - sourceX) + (j - sourceY) * (j - sourceY);
				float dist = Mathf.Sqrt(distSq);

				// Only add density within radius
				if (dist <= radiusInCells)
				{
					// Reduce strength as farther from center (linear falloff)
					float falloff = 1.0f - (dist / radiusInCells);
					AddDensity(i, j, effectiveStrength * falloff);

					// Add velocity if enabled
					if (sourceEmitsVelocity)
					{
						// Convert direction angle to velocity components
						float angleRad = sourceDirection * Mathf.Deg2Rad;
						float vx = Mathf.Cos(angleRad) * sourceVelocity * resolutionMultiplier * falloff;
						float vy = Mathf.Sin(angleRad) * sourceVelocity * resolutionMultiplier * falloff;

						AddVelocity(i, j, vx, vy);
					}
				}
			}
		}
	}

	Vector2 GetMousePositionInGrid()
	{
		Vector3 mouseScreenPos = Input.mousePosition;
		Vector3 mouseWorldPos = mainCamera.ScreenToWorldPoint(new Vector3(mouseScreenPos.x, mouseScreenPos.y, -mainCamera.transform.position.z));

		float minX = Mathf.Min(quadCorners[0].x, quadCorners[1].x, quadCorners[2].x, quadCorners[3].x);
		float maxX = Mathf.Max(quadCorners[0].x, quadCorners[1].x, quadCorners[2].x, quadCorners[3].x);
		float minY = Mathf.Min(quadCorners[0].y, quadCorners[1].y, quadCorners[2].y, quadCorners[3].y);
		float maxY = Mathf.Max(quadCorners[0].y, quadCorners[1].y, quadCorners[2].y, quadCorners[3].y);

		float normalizedX = (mouseWorldPos.x - minX) / (maxX - minX);
		float normalizedY = (mouseWorldPos.y - minY) / (maxY - minY);

		return new Vector2(normalizedX * currentSize, normalizedY * currentSize);
	}

	void Simulate()
	{
		// Scale diffusion and viscosity with resolution
		float effectiveTimeStep = autoAdjustParameters ? timeStep * dtScale : timeStep;
		float effectiveDiffusion = autoAdjustParameters ? diffusion / resolutionMultiplier : diffusion;
		float effectiveViscosity = autoAdjustParameters ? viscosity / resolutionMultiplier : viscosity;

		VelocityStep(effectiveTimeStep, effectiveViscosity);
		DensityStep(effectiveTimeStep, effectiveDiffusion);

		if (applyTurbulentNoise)
		{
			ApplyTurbulentNoise();
		}

		// Apply obstacle interactions
		if (enableObstacle)
		{
			EnforceObstacleBoundaries();
		}

		if (_enableRuntimeLogging && currentStep % _loggingInterval == 0)
		{
			LogCurrentMetrics();
		}
	}

	private void LogCurrentMetrics()
	{
		if (_currentRunID == -1) return;

		float avgDensity = 0f;
		float maxVelocity = 0f;
		float frameRate = CalculateFrameRate();

		// Calculate metrics
		foreach (float d in density) avgDensity += d;
		avgDensity /= density.Length;

		for (int i = 0; i < velocityX.Length; i++)
		{
			float mag = Mathf.Sqrt(velocityX[i] * velocityX[i] + velocityY[i] * velocityY[i]);
			if (mag > maxVelocity) maxVelocity = mag;
		}

		// Log to SQL
		if (maxVelocity != 0 && avgDensity != 0)
		{
			SQL.LogRuntimeMetrics(
			_currentRunID,
			currentStep,
			avgDensity,
			maxVelocity,
			frameRate
		);
		}
	}

	private float CalculateFrameRate()
	{
		float instantFPS = 1.0f / Time.unscaledDeltaTime;
		_smoothedFPS = SMOOTH_FACTOR * _smoothedFPS +
					   (1 - SMOOTH_FACTOR) * instantFPS;
		return _smoothedFPS;
	}

	void EnforceObstacleBoundaries()
	{
		// Ensure zero velocity inside obstacles
		for (int i = 1; i < currentSize - 1; i++)
		{
			for (int j = 1; j < currentSize - 1; j++)
			{
				int idx = GridUtils.IX(i, j, currentSize);
				if (obstacles[idx])
				{
					velocityX[idx] = obstacles[idx] ? 0 : velocityX[idx];
					velocityY[idx] = obstacles[idx] ? 0 : velocityY[idx];

					ApplyDragNearObstacle(i, j);
				}
			}
		}
	}

	void ApplyDragNearObstacle(int obstacleI, int obstacleJ)
	{
		// Characteristic length (grid cell size)
		float L = cellSize;

		// Check and apply to all adjacent cells
		int[] di = { -1, 1, 0, 0 };
		int[] dj = { 0, 0, -1, 1 };

		for (int n = 0; n < 4; n++)
		{
			int ni = obstacleI + di[n];
			int nj = obstacleJ + dj[n];

			// Skip if out of bounds
			if (ni < 1 || ni >= currentSize - 1 || nj < 1 || nj >= currentSize - 1)
				continue;

			int nidx = GridUtils.IX(ni, nj, currentSize);

			// Skip if this is also an obstacle
			if (obstacles[nidx])
				continue;

			// Compute velocity magnitude
			float U = Mathf.Sqrt(velocityX[nidx] * velocityX[nidx] + velocityY[nidx] * velocityY[nidx]);

			// Compute Reynolds number (avoid division by zero)
			float Re = (U * L) / Mathf.Max(viscosity, 1e-5f);

			// Adapt drag factor based on Reynolds number (sigmoid-like response)
			float dragFactor = Mathf.Lerp(0.8f, 0.98f, 1.0f - Mathf.Exp(-Re * 0.01f));

			// Apply adaptive drag
			velocityX[nidx] *= dragFactor;
			velocityY[nidx] *= dragFactor;
		}
	}

	void ApplyTurbulentNoise()
	{
		float noiseScale = 0.1f;  // Adjusts turbulence intensity
		float frequency = 0.05f;  // Controls size of turbulent structures

		for (int i = 1; i < currentSize - 1; i++)
		{
			for (int j = 1; j < currentSize - 1; j++)
			{
				int idx = GridUtils.IX(i, j, currentSize);

				// Compute velocity magnitude
				float U = Mathf.Sqrt(velocityX[idx] * velocityX[idx] + velocityY[idx] * velocityY[idx]);

				// Generate Perlin noise (scaled by frequency)
				float noiseX = Mathf.PerlinNoise(i * frequency, j * frequency) - 0.5f;
				float noiseY = Mathf.PerlinNoise(j * frequency, i * frequency) - 0.5f;

				// Scale perturbation by velocity magnitude
				float perturbationStrength = noiseScale * U;

				// Apply turbulence to velocity
				velocityX[idx] += noiseX * perturbationStrength;
				velocityY[idx] += noiseY * perturbationStrength;
			}
		}
	}

	void VelocityStep(float dt, float visc)
	{
		Diffuse(1, velocityX0, velocityX, visc, dt);
		Diffuse(2, velocityY0, velocityY, visc, dt);

		ProjectWithJobs(velocityX0, velocityY0, velocityX, velocityY);

		AdvectWithJobs(1, velocityX, velocityX0, velocityX0, velocityY0, dt);
		AdvectWithJobs(2, velocityY, velocityY0, velocityX0, velocityY0, dt);

		ProjectWithJobs(velocityX, velocityY, velocityX0, velocityY0);
	}

	void DensityStep(float dt, float diff)
	{
		float[] densityTemp = new float[currentSize * currentSize];
		Diffuse(0, densityTemp, density, diff, dt);
		AdvectWithJobs(0, density, densityTemp, velocityX, velocityY, dt);
	}

	void AddDensity(float x, float y, float amount)
	{
		int i = Mathf.Clamp((int)x, 0, currentSize - 1);
		int j = Mathf.Clamp((int)y, 0, currentSize - 1);

		density[GridUtils.IX(i, j, currentSize)] += amount;
	}

	void AddVelocity(float x, float y, float amountX, float amountY)
	{
		int i = Mathf.Clamp((int)x, 0, currentSize - 1);
		int j = Mathf.Clamp((int)y, 0, currentSize - 1);

		velocityX[GridUtils.IX(i, j, currentSize)] += amountX;
		velocityY[GridUtils.IX(i, j, currentSize)] += amountY;
	}

	void Diffuse(int b, float[] x, float[] x0, float diff, float dt)
	{
		DiffuseWithJobs(b, x, x0, diff, dt);
		float a = dt * diff * (currentSize - 2) * (currentSize - 2);
		LinearSolveWithJobs(b, x, x0, a, 1 + 4 * a);
	}

	public static class GridUtils
	{
		public static int IX(int x, int y, int size)
		{
			return x + y * size;
		}
	}

	void UpdateVisualization()
	{
		//NativeArray initialisation
		Color[] colours = new Color[currentSize * currentSize];
		NativeArray<Color> nativeColors = new NativeArray<Color>(currentSize * currentSize, Allocator.TempJob);

		NativeArray<float> nativeDensity = new NativeArray<float>(density, Allocator.TempJob);
		NativeArray<bool> nativeObstacles = new NativeArray<bool>(obstacles, Allocator.TempJob);

		NativeArray<Color> gradientColors = new NativeArray<Color>(1, Allocator.TempJob);
		NativeArray<float> gradientTimes = new NativeArray<float>(1, Allocator.TempJob);
		int gradientKeyCount = 0;

		NativeArray<float> nativePressure = new NativeArray<float>(pressure, Allocator.TempJob);

		if (colorMode == ColorMode.Gradient && colourGradient != null)
		{
			GradientColorKey[] colorKeys = colourGradient.colorKeys;
			gradientKeyCount = colorKeys.Length;

			// Dispose the initial arrays and create new ones with the right size
			gradientColors.Dispose();
			gradientTimes.Dispose();

			gradientColors = new NativeArray<Color>(gradientKeyCount, Allocator.TempJob);
			gradientTimes = new NativeArray<float>(gradientKeyCount, Allocator.TempJob);

			for (int i = 0; i < gradientKeyCount; i++)
			{
				gradientColors[i] = colorKeys[i].color;
				gradientTimes[i] = colorKeys[i].time;
			}
		}

		// colour change over time
		if (useLerp != false)
		{
			float cycle = Mathf.PingPong(elapsedTime * 0.1f, 1f);
			fluidcolour = Color.Lerp(startColor, endColor, cycle);
		}

		try
		{
			// Create and schedule the job
			var visualizationJob = new UpdateVisualizationJob
			{
				colors = nativeColors,
				density = nativeDensity,
				obstacles = nativeObstacles,
				size = currentSize,
				sourceX = sourcePositionX * currentSize,
				sourceY = sourcePositionY * currentSize,
				visualMarkerRadius = 3, // Size of the position marker in pixels
				colorMode = colorMode,
				fluidColor = fluidcolour,
				obstacleColor = obstacleColor,
				sourcePositionColor = sourcePositionColor,
				colourIntensity = colourIntensity,
				visualizeSourcePosition = visualizeSourcePosition,
				enableCustomSource = enableCustomSource,
				mediumDensityThreshold = mediumDensityThreshold,
				highDensityThreshold = highDensityThreshold,
				lowDensityColor = lowDensityColor,
				mediumDensityColor = mediumDensityColor,
				highDensityColor = highDensityColor,
				gradientColors = gradientColors,
				gradientTimes = gradientTimes,
				gradientKeyCount = gradientKeyCount,
				pressure = nativePressure,
				lowPressureColor = lowPressureColor,
				neutralPressureColor = neutralPressureColor,
				highPressureColor = highPressureColor,
				lowPressureThreshold = lowPressureThreshold,
				highPressureThreshold = highPressureThreshold
			};

			// Schedule the job with one item per cell
			JobHandle jobHandle = visualizationJob.Schedule(currentSize * currentSize, 64);

			// Wait for the job to complete
			jobHandle.Complete();

			// Copy the results back to the managed array
			nativeColors.CopyTo(colours);
		}
		finally
		{
			// Clean up native arrays
			nativeColors.Dispose();
			nativeDensity.Dispose();
			nativeObstacles.Dispose();
			gradientColors.Dispose();
			gradientTimes.Dispose();
			nativePressure.Dispose();
		}

		// Apply the colors to the texture
		fluidTexture.SetPixels(colours);
		fluidTexture.Apply();

		// If streamlines are enabled, draw them
		if (showStreamlines || colorMode == ColorMode.Streamlines)
		{
			DrawStreamlines();
		}

		// If in streamline mode, combine the textures
		if (colorMode == ColorMode.Streamlines)
		{
			CombineTextures();
		}
	}

	void CombineTextures()
	{
		Color[] fluidColors = fluidTexture.GetPixels();
		Color[] streamColors = streamlineTexture.GetPixels();

		for (int i = 0; i < fluidColors.Length; i++)
		{
			// Only draw streamline pixels that are not transparent
			if (streamColors[i].a > 0)
			{
				fluidColors[i] = streamColors[i];
			}
		}

		fluidTexture.SetPixels(fluidColors);
		fluidTexture.Apply();
	}

	void DrawStreamlines()
	{
		// Only process if streamline visualization is enabled
		if (!showStreamlines && colorMode != ColorMode.Streamlines) return;

		// Calculate the skip value based on density
		int skip = Mathf.Max(1, currentSize / (streamlineDensity * 10));

		// Calculate how many streamlines needed to process
		int streamlineCount = (currentSize / skip) * (currentSize / skip);

		// Initialize the streamline texture if needed
		if (streamlineTexture == null || streamlineTexture.width != currentSize)
		{
			if (streamlineTexture != null) Destroy(streamlineTexture);
			streamlineTexture = new Texture2D(currentSize, currentSize, TextureFormat.RGBA32, false);
			streamlineTexture.filterMode = FilterMode.Point;
		}

		// Initialize or resize job arrays if needed
		InitializeStreamlineBuffers(streamlineCount);

		// Get texture colors for processing
		Color[] colors = new Color[currentSize * currentSize];

		// Create a native array for line segments
		NativeArray<float4> lineSegments = new NativeArray<float4>(streamlineCount, Allocator.TempJob);

		try
		{
			// Calculate streamlines in parallel
			var calculationJob = new StreamlineCalculationJob
			{
				velocX = new NativeArray<float>(velocityX, Allocator.TempJob),
				velocY = new NativeArray<float>(velocityY, Allocator.TempJob),
				streamlines = streamlineData,
				obstacles = new NativeArray<bool>(obstacles, Allocator.TempJob),
				size = currentSize,
				skip = skip,
				streamlineScale = streamlineScale
			};

			JobHandle calcHandle = calculationJob.Schedule(streamlineCount, 32);

			// Generate line segments
			var lineSegmentJob = new StreamlineDrawJob
			{
				streamlines = streamlineData,
				lineSegments = lineSegments,
				textureSize = currentSize,
				halfThickness = streamlineThickness / 2f
			};

			JobHandle lineSegmentHandle = lineSegmentJob.Schedule(streamlineCount, 16, calcHandle);
			lineSegmentHandle.Complete();

			// Draw all line segments on CPU (no race conditions)
			DrawLineSegmentsToTexture(lineSegments, colors);

			// Clean up the temporary arrays
			calculationJob.velocX.Dispose();
			calculationJob.velocY.Dispose();
			calculationJob.obstacles.Dispose();
		}
		finally
		{
			// Ensure always dispose the temp native array
			lineSegments.Dispose();
		}

		// Apply colors to the texture
		streamlineTexture.SetPixels(colors);
		streamlineTexture.Apply();
	}

	void InitializeStreamlineBuffers(int streamlineCount)
	{
		if (streamlineBuffersInitialized && streamlineData.Length == streamlineCount)
			return;

		// Clean up existing buffers if they exist
		if (streamlineBuffersInitialized)
		{
			streamlineData.Dispose();
			streamlineBuffersInitialized = false;
		}

		// Create new buffers
		streamlineData = new NativeArray<float4>(streamlineCount, Allocator.Persistent);
		streamlineBuffersInitialized = true;
	}

	// Helper method to get the current source position	
	public Vector2 GetSourcePosition()
	{
		return new Vector2(sourcePositionX * currentSize, sourcePositionY * currentSize);
	}

	public void SetSourcePosition(float x, float y)
	{
		sourcePositionX = Mathf.Clamp01(x / currentSize);
		sourcePositionY = Mathf.Clamp01(y / currentSize);
	}

	private void InitializeJobBuffers()
	{
		if (jobBuffersInitialized)
			return;

		int totalSize = currentSize * currentSize;
		jobBuffer1 = new NativeArray<float>(totalSize, Allocator.Persistent);
		jobBuffer2 = new NativeArray<float>(totalSize, Allocator.Persistent);
		jobObstacles = new NativeArray<bool>(totalSize, Allocator.Persistent);
		jobBuffersInitialized = true;
	}

	void OnDestroy()
	{
		if (jobBuffersInitialized)
		{
			jobBuffer1.Dispose();
			jobBuffer2.Dispose();
			jobObstacles.Dispose();
			jobBuffersInitialized = false;
		}

		// Clean up streamline buffers
		if (streamlineBuffersInitialized)
		{
			streamlineData.Dispose();
			streamlineBuffersInitialized = false;
		}
	}

	void ResetJobBuffers()
	{
		if (jobBuffersInitialized)
		{
			jobBuffer1.Dispose();
			jobBuffer2.Dispose();
			jobObstacles.Dispose();
			jobBuffersInitialized = false;
		}

		// Reinitialize with new size
		InitializeJobBuffers();
	}

	[BurstCompile]
	public struct DiffuseJob : IJobParallelFor
	{
		[ReadOnly] public NativeArray<float> input;
		public NativeArray<float> output;
		[ReadOnly] public NativeArray<bool> obstacles;

		public int size;
		public float a;
		public float c;

		public void Execute(int index)
		{
			// Skip boundary cells
			int i = index % size;
			int j = index / size;

			if (i <= 0 || i >= size - 1 || j <= 0 || j >= size - 1)
				return;

			// Skip obstacles
			if (obstacles[index])
				return;
			int p = size;
			// IX function inlined
			int getIndex(int x, int y) => x + y * p;

			// Read from input, write to output - no race condition
			output[index] = (input[index] + a * (
				input[getIndex(i + 1, j)] +
				input[getIndex(i - 1, j)] +
				input[getIndex(i, j + 1)] +
				input[getIndex(i, j - 1)]
			)) / c;
		}
	}

	[BurstCompile]
	public struct ProjectDivergenceJob : IJobParallelFor
	{
		[WriteOnly] public NativeArray<float> div;
		[WriteOnly] public NativeArray<float> p;
		[ReadOnly] public NativeArray<float> velocX;
		[ReadOnly] public NativeArray<float> velocY;
		public int size;

		public void Execute(int index)
		{
			// Skip boundary cells
			int i = index % size;
			int j = index / size;

			if (i <= 0 || i >= size - 1 || j <= 0 || j >= size - 1)
				return;

			div[index] = -0.5f * (
				velocX[GridUtils.IX(i + 1, j, size)] - velocX[GridUtils.IX(i - 1, j, size)] +
				velocY[GridUtils.IX(i, j + 1, size)] - velocY[GridUtils.IX(i, j - 1, size)]
			) / size;

			p[index] = 0;
		}
	}

	[BurstCompile]
	public struct ProjectVelocityAdjustJob : IJobParallelFor
	{
		public NativeArray<float> velocX;
		public NativeArray<float> velocY;
		[ReadOnly] public NativeArray<float> p;
		[ReadOnly] public NativeArray<bool> obstacles;
		public int size;

		public void Execute(int index)
		{
			// Skip boundary cells
			int i = index % size;
			int j = index / size;

			if (i <= 0 || i >= size - 1 || j <= 0 || j >= size - 1)
				return;

			// Skip obstacles
			if (obstacles[index])
				return;

			velocX[index] -= 0.5f * (p[GridUtils.IX(i + 1, j, size)] - p[GridUtils.IX(i - 1, j, size)]) * size;
			velocY[index] -= 0.5f * (p[GridUtils.IX(i, j + 1, size)] - p[GridUtils.IX(i, j - 1, size)]) * size;
		}
	}

	[BurstCompile]
	public struct AdvectJob : IJobParallelFor
	{
		[ReadOnly] public NativeArray<float> d0;
		[WriteOnly] public NativeArray<float> d;
		[ReadOnly] public NativeArray<float> velocX;
		[ReadOnly] public NativeArray<float> velocY;
		[ReadOnly] public NativeArray<bool> obstacles;

		public int size;
		public int b;
		public float dt0;

		public void Execute(int index)
		{
			// Skip boundary cells
			int i = index % size;
			int j = index / size;

			if (i <= 0 || i >= size - 1 || j <= 0 || j >= size - 1)
				return;

			// Skip obstacle cells for velocity fields
			if (obstacles[index] && (b == 1 || b == 2))
			{
				d[index] = 0;
				return;
			}

			// Skip obstacles for density (leave unchanged)
			if (obstacles[index])
				return;

			float x = i - dt0 * velocX[index];
			float y = j - dt0 * velocY[index];

			// Clamp values to grid boundaries
			if (x < 0.5f) x = 0.5f;
			if (x > size - 1.5f) x = size - 1.5f;
			int i0 = (int)x;
			int i1 = i0 + 1;

			if (y < 0.5f) y = 0.5f;
			if (y > size - 1.5f) y = size - 1.5f;
			int j0 = (int)y;
			int j1 = j0 + 1;

			// Bilinear interpolation weights
			float s1 = x - i0;
			float s0 = 1 - s1;
			float t1 = y - j0;
			float t0 = 1 - t1;


			int s = size;
			int getIndex(int x, int y) => x + y * s;

			// Bilinear interpolation
			d[index] = s0 * (t0 * d0[getIndex(i0, j0)] + t1 * d0[getIndex(i0, j1)]) +
					   s1 * (t0 * d0[getIndex(i1, j0)] + t1 * d0[getIndex(i1, j1)]);
		}
	}

	[BurstCompile]
	public struct LinearSolveIterationJob : IJobParallelFor
	{
		[ReadOnly] public NativeArray<float> x0;
		[ReadOnly] public NativeArray<float> xRead;  // Read from previous iteration
		[WriteOnly] public NativeArray<float> xWrite; // Write to next iteration
		[ReadOnly] public NativeArray<bool> obstacles;

		public int size;
		public float a;
		public float c;

		public void Execute(int index)
		{
			// Skip boundary cells
			int i = index % size;
			int j = index / size;

			if (i <= 0 || i >= size - 1 || j <= 0 || j >= size - 1)
			{
				// Just copy the value for boundary cells
				xWrite[index] = xRead[index];
				return;
			}

			// Skip obstacles (just copy the value)
			if (obstacles[index])
			{
				xWrite[index] = xRead[index];
				return;
			}

			// Use proper indexing function to get surrounding values
			int left = GridUtils.IX(i - 1, j, size);
			int right = GridUtils.IX(i + 1, j, size);
			int top = GridUtils.IX(i, j + 1, size);
			int bottom = GridUtils.IX(i, j - 1, size);

			// LinearSolve computation
			xWrite[index] = (x0[index] + a * (
				xRead[right] + xRead[left] +
				xRead[top] + xRead[bottom]
			)) / c;
		}

	}

	[BurstCompile]
	public struct BoundaryJob : IJob
	{
		public NativeArray<float> x;
		[ReadOnly] public NativeArray<bool> obstacles;
		public int size;
		public int b; // Boundary condition flag

		public void Execute()
		{
			// Handle outer grid boundaries
			for (int i = 1; i < size - 1; i++)
			{
				x[GridUtils.IX(0, i, size)] = b == 1 ? -x[GridUtils.IX(1, i, size)] : x[GridUtils.IX(1, i, size)];
				x[GridUtils.IX(size - 1, i, size)] = b == 1 ? -x[GridUtils.IX(size - 2, i, size)] : x[GridUtils.IX(size - 2, i, size)];
				x[GridUtils.IX(i, 0, size)] = b == 2 ? -x[GridUtils.IX(i, 1, size)] : x[GridUtils.IX(i, 1, size)];
				x[GridUtils.IX(i, size - 1, size)] = b == 2 ? -x[GridUtils.IX(i, size - 2, size)] : x[GridUtils.IX(i, size - 2, size)];
			}

			// Corner cells
			x[GridUtils.IX(0, 0, size)] = 0.5f * (x[GridUtils.IX(1, 0, size)] + x[GridUtils.IX(0, 1, size)]);
			x[GridUtils.IX(0, size - 1, size)] = 0.5f * (x[GridUtils.IX(1, size - 1, size)] + x[GridUtils.IX(0, size - 2, size)]);
			x[GridUtils.IX(size - 1, 0, size)] = 0.5f * (x[GridUtils.IX(size - 2, 0, size)] + x[GridUtils.IX(size - 1, 1, size)]);
			x[GridUtils.IX(size - 1, size - 1, size)] = 0.5f * (x[GridUtils.IX(size - 2, size - 1, size)] + x[GridUtils.IX(size - 1, size - 2, size)]);

			// Handle internal obstacle boundaries with mirroring
			for (int i = 1; i < size - 1; i++)
			{
				for (int j = 1; j < size - 1; j++)
				{
					int idx = GridUtils.IX(i, j, size);
					if (obstacles[idx])
					{
						// Mirror velocity from the nearest fluid cells
						if (b == 1) // Horizontal velocity (velocityX)
						{
							float mirroredX = 0;
							int count = 0;
							if (!obstacles[GridUtils.IX(i - 1, j, size)]) { mirroredX += -x[GridUtils.IX(i - 1, j, size)]; count++; }
							if (!obstacles[GridUtils.IX(i + 1, j, size)]) { mirroredX += -x[GridUtils.IX(i + 1, j, size)]; count++; }
							x[idx] = count > 0 ? mirroredX / count : 0;
						}
						else if (b == 2) // Vertical velocity (velocityY)
						{
							float mirroredY = 0;
							int count = 0;
							if (!obstacles[GridUtils.IX(i, j - 1, size)]) { mirroredY += -x[GridUtils.IX(i, j - 1, size)]; count++; }
							if (!obstacles[GridUtils.IX(i, j + 1, size)]) { mirroredY += -x[GridUtils.IX(i, j + 1, size)]; count++; }
							x[idx] = count > 0 ? mirroredY / count : 0;
						}
					}
				}
			}
		}
	}


	void DiffuseWithJobs(int b, float[] x, float[] x0, float diff, float dt)
	{
		int totalSize = currentSize * currentSize;
		float a = dt * diff * (currentSize - 2) * (currentSize - 2);
		float c = 1 + 4 * a;

		// Create native arrays for double buffering
		NativeArray<float> buffer1 = new NativeArray<float>(x0, Allocator.TempJob);
		NativeArray<float> buffer2 = new NativeArray<float>(x0, Allocator.TempJob);
		NativeArray<bool> nativeObstacles = new NativeArray<bool>(obstacles, Allocator.TempJob);

		try
		{
			// Set up which buffer is input/output
			NativeArray<float> input = buffer1;
			NativeArray<float> output = buffer2;

			// Multiple iterations of the solver
			for (int k = 0; k < 20; k++)
			{
				// Create the diffusion job
				var diffuseJob = new DiffuseJob
				{
					input = input,
					output = output,
					obstacles = nativeObstacles,
					size = currentSize,
					a = a,
					c = c
				};

				// Schedule the parallel job
				JobHandle jobHandle = diffuseJob.Schedule(totalSize, 64);

				// Create boundary job to run after diffusion
				var boundaryJob = new BoundaryJob
				{
					x = output,
					obstacles = nativeObstacles,
					size = currentSize,
					b = b
				};

				// Schedule boundary job with dependency on diffusion
				JobHandle boundaryHandle = boundaryJob.Schedule(jobHandle);

				// Wait for completion before next iteration
				boundaryHandle.Complete();

				// Swap buffers for next iteration
				var temp = input;
				input = output;
				output = temp;
			}

			// Final result is in the input buffer after the last swap
			input.CopyTo(x);
		}
		finally
		{
			// Clean up native arrays
			buffer1.Dispose();
			buffer2.Dispose();
			nativeObstacles.Dispose();
		}
	}

	void LinearSolveWithJobs(int b, float[] x, float[] x0, float a, float c)
	{
		int totalSize = currentSize * currentSize;

		// Ensure buffers are initialized
		InitializeJobBuffers();

		// Copy input data to native arrays
		jobBuffer1.CopyFrom(x);
		NativeArray<float> tempX0 = new NativeArray<float>(x0, Allocator.TempJob);
		jobObstacles.CopyFrom(obstacles);

		try
		{
			// Set up input and output buffers for double buffering
			NativeArray<float> readBuffer = jobBuffer1;
			NativeArray<float> writeBuffer = jobBuffer2;

			// Run multiple iterations of the solver
			for (int k = 0; k < 20; k++)
			{
				// Create the linear solve job
				var linearSolveJob = new LinearSolveIterationJob
				{
					x0 = tempX0,
					xRead = readBuffer,
					xWrite = writeBuffer,
					obstacles = jobObstacles,
					size = currentSize,
					a = a,
					c = c
				};

				// Schedule the parallel job
				JobHandle jobHandle = linearSolveJob.Schedule(totalSize, 64);

				// Wait for the job to complete
				jobHandle.Complete();

				// Apply boundary conditions
				ApplyBoundaryConditions(b, writeBuffer);

				// Swap buffers for next iteration
				var temp = readBuffer;
				readBuffer = writeBuffer;
				writeBuffer = temp;
			}

			// Copy results back to the original array (from readBuffer after the last swap)
			readBuffer.CopyTo(x);
		}
		finally
		{
			// Only dispose the temporary x0 array
			tempX0.Dispose();
		}
	}

	void ProjectWithJobs(float[] velocX, float[] velocY, float[] p, float[] div)
	{
		int totalSize = currentSize * currentSize;

		// Ensure job buffers are initialized
		InitializeJobBuffers();

		// Create native arrays for job data
		NativeArray<float> nativeVelocX = new NativeArray<float>(velocX, Allocator.TempJob);
		NativeArray<float> nativeVelocY = new NativeArray<float>(velocY, Allocator.TempJob);
		NativeArray<float> nativeP = new NativeArray<float>(totalSize, Allocator.TempJob);
		NativeArray<float> nativeDiv = new NativeArray<float>(totalSize, Allocator.TempJob);
		NativeArray<bool> nativeObstacles = new NativeArray<bool>(obstacles, Allocator.TempJob);

		try
		{
			// Calculate divergence
			var divergenceJob = new ProjectDivergenceJob
			{
				div = nativeDiv,
				p = nativeP,
				velocX = nativeVelocX,
				velocY = nativeVelocY,
				size = currentSize
			};

			JobHandle divergenceHandle = divergenceJob.Schedule(totalSize, 64);

			// Apply boundary conditions (dependent on divergence calculation)
			var boundaryDiv = new BoundaryJob
			{
				x = nativeDiv,
				obstacles = nativeObstacles,
				size = currentSize,
				b = 0
			};

			var boundaryP = new BoundaryJob
			{
				x = nativeP,
				obstacles = nativeObstacles,
				size = currentSize,
				b = 0
			};

			JobHandle boundaryDivHandle = boundaryDiv.Schedule(divergenceHandle);
			JobHandle boundaryPHandle = boundaryP.Schedule(divergenceHandle);

			// Wait for both boundary operations to complete
			JobHandle.CombineDependencies(boundaryDivHandle, boundaryPHandle).Complete();

			PressureSolveWithJobs(nativeP, nativeDiv, nativeObstacles);

			// Adjust velocities
			var velocityJob = new ProjectVelocityAdjustJob
			{
				velocX = nativeVelocX,
				velocY = nativeVelocY,
				p = nativeP,
				obstacles = nativeObstacles,
				size = currentSize
			};

			JobHandle velocityHandle = velocityJob.Schedule(totalSize, 64);

			//  Apply boundary conditions to velocities
			var boundaryVelocX = new BoundaryJob
			{
				x = nativeVelocX,
				obstacles = nativeObstacles,
				size = currentSize,
				b = 1
			};

			var boundaryVelocY = new BoundaryJob
			{
				x = nativeVelocY,
				obstacles = nativeObstacles,
				size = currentSize,
				b = 2
			};

			JobHandle boundaryVelocXHandle = boundaryVelocX.Schedule(velocityHandle);
			JobHandle boundaryVelocYHandle = boundaryVelocY.Schedule(velocityHandle);

			// Wait for both boundary operations to complete
			JobHandle.CombineDependencies(boundaryVelocXHandle, boundaryVelocYHandle).Complete();

			// Copy results back to managed arrays
			nativeVelocX.CopyTo(velocX);
			nativeVelocY.CopyTo(velocY);
			nativeP.CopyTo(p);
			nativeP.CopyTo(pressure);

		}
		finally
		{
			// Clean up native arrays
			nativeVelocX.Dispose();
			nativeVelocY.Dispose();
			nativeP.Dispose();
			nativeDiv.Dispose();
			nativeObstacles.Dispose();
		}
	}

	void AdvectWithJobs(int b, float[] d, float[] d0, float[] velocX, float[] velocY, float dt)
	{
		int totalSize = currentSize * currentSize;
		float dt0 = dt * (currentSize - 2);

		// Create native arrays for job data
		NativeArray<float> nativeD = new NativeArray<float>(totalSize, Allocator.TempJob);
		NativeArray<float> nativeD0 = new NativeArray<float>(d0, Allocator.TempJob);
		NativeArray<float> nativeVelocX = new NativeArray<float>(velocX, Allocator.TempJob);
		NativeArray<float> nativeVelocY = new NativeArray<float>(velocY, Allocator.TempJob);
		NativeArray<bool> nativeObstacles = new NativeArray<bool>(obstacles, Allocator.TempJob);

		try
		{
			// Create and schedule the advection job
			var advectJob = new AdvectJob
			{
				d = nativeD,
				d0 = nativeD0,
				velocX = nativeVelocX,
				velocY = nativeVelocY,
				obstacles = nativeObstacles,
				size = currentSize,
				b = b,
				dt0 = dt0
			};

			JobHandle advectHandle = advectJob.Schedule(totalSize, 64);

			// Apply boundary conditions afterward
			var boundaryJob = new BoundaryJob
			{
				x = nativeD,
				obstacles = nativeObstacles,
				size = currentSize,
				b = b
			};

			JobHandle boundaryHandle = boundaryJob.Schedule(advectHandle);
			boundaryHandle.Complete();

			// Copy results back to managed array
			nativeD.CopyTo(d);
		}
		finally
		{
			// Clean up native arrays
			nativeD.Dispose();
			nativeD0.Dispose();
			nativeVelocX.Dispose();
			nativeVelocY.Dispose();
			nativeObstacles.Dispose();
		}
	}

	void PressureSolveWithJobs(NativeArray<float> p, NativeArray<float> div, NativeArray<bool> obstacles)
	{
		// Similar to LinearSolveWithJobs but specifically for pressure calculation
		float a = 1.0f;
		float c = 6.0f;
		int totalSize = p.Length;

		// Create temporary buffer for double buffering
		NativeArray<float> tempBuffer = new NativeArray<float>(totalSize, Allocator.TempJob);

		try
		{
			NativeArray<float> readBuffer = p;
			NativeArray<float> writeBuffer = tempBuffer;

			// Run multiple iterations of the solver
			for (int k = 0; k < 20; k++)
			{
				var linearSolveJob = new LinearSolveIterationJob
				{
					x0 = div,
					xRead = readBuffer,
					xWrite = writeBuffer,
					obstacles = obstacles,
					size = currentSize,
					a = a,
					c = c
				};

				JobHandle jobHandle = linearSolveJob.Schedule(totalSize, 64);
				jobHandle.Complete();

				// Apply boundary conditions
				var boundaryJob = new BoundaryJob
				{
					x = writeBuffer,
					obstacles = obstacles,
					size = currentSize,
					b = 0
				};

				boundaryJob.Schedule().Complete();

				// Swap buffers for next iteration
				var temp = readBuffer;
				readBuffer = writeBuffer;
				writeBuffer = temp;
			}

			// If the final result is in tempBuffer, copy it to p
			if (readBuffer.GetHashCode() != p.GetHashCode())
			{
				readBuffer.CopyTo(p);
			}
		}
		finally
		{
			tempBuffer.Dispose();
		}
	}

	void ApplyBoundaryConditions(int b, NativeArray<float> buffer)
	{
		// Create and execute boundary job
		var boundaryJob = new BoundaryJob
		{
			x = buffer,
			obstacles = new NativeArray<bool>(obstacles, Allocator.TempJob),
			size = currentSize,
			b = b
		};

		JobHandle boundaryHandle = boundaryJob.Schedule();
		boundaryHandle.Complete();

		// Clean up the temporary obstacles array
		boundaryJob.obstacles.Dispose();
	}

	[BurstCompile]
	public struct ClearTextureJob : IJobParallelFor
	{
		[WriteOnly] public NativeArray<Color> colors;

		public void Execute(int index)
		{
			colors[index] = new Color(0, 0, 0, 0);
		}
	}

	[BurstCompile]
	public struct StreamlineCalculationJob : IJobParallelFor
	{
		[ReadOnly] public NativeArray<float> velocX;
		[ReadOnly] public NativeArray<float> velocY;
		[WriteOnly] public NativeArray<float4> streamlines;
		[ReadOnly] public NativeArray<bool> obstacles;

		public int size;
		public int skip;
		public float streamlineScale;

		public void Execute(int index)
		{
			// Calculate 2D position from index
			int x = index % (size / skip);
			int y = index / (size / skip);

			// Convert to actual grid coordinates
			int i = x * skip + skip;
			int j = y * skip + skip;

			// Skip if outside the valid range
			if (i <= 0 || i >= size - 1 || j <= 0 || j >= size - 1)
			{
				streamlines[index] = new float4(i, j, 0, 0); // Mark as invalid
				return;
			}

			int idx = GridUtils.IX(i, j, size);

			// Skip obstacles
			if (obstacles[idx])
			{
				streamlines[index] = new float4(i, j, 0, 0); // Mark as invalid
				return;
			}

			float vx = velocX[idx];
			float vy = velocY[idx];

			// Calculate velocity magnitude
			float magnitude = math.sqrt(vx * vx + vy * vy);

			// Skip cells with very little flow
			if (magnitude < 0.01f)
			{
				streamlines[index] = new float4(i, j, 0, 0); // Mark as invalid
				return;
			}

			// Calculate line length based on velocity magnitude and scale
			float lineLength = math.min(skip - 1, magnitude * streamlineScale);

			// Calculate the angle of the velocity
			float angle = math.atan2(vy, vx);

			// Store the streamline information
			streamlines[index] = new float4(i, j, angle, lineLength);
		}
	}

	[BurstCompile]
	public struct StreamlineDrawJob : IJobParallelFor
	{
		[ReadOnly] public NativeArray<float4> streamlines;
		// create line segments
		[WriteOnly] public NativeArray<float4> lineSegments; // x,y,endX,endY
		public int textureSize;
		public float halfThickness;

		public void Execute(int index)
		{
			float4 streamline = streamlines[index];

			// Skip invalid streamlines
			if (streamline.w <= 0)
			{
				// Store a "null" line segment
				lineSegments[index] = new float4(-1, -1, -1, -1);
				return;
			}

			int startX = (int)streamline.x;
			int startY = (int)streamline.y;
			float angle = streamline.z;
			float length = streamline.w;

			// Calculate the endpoint
			float endX = startX + math.cos(angle) * length;
			float endY = startY + math.sin(angle) * length;

			// Store line segment data
			lineSegments[index] = new float4(startX, startY, endX, endY);
		}
	}

	private void DrawLineSegmentsToTexture(NativeArray<float4> lineSegments, Color[] textureColors)
	{
		for (int i = 0; i < lineSegments.Length; i++)
		{
			float4 segment = lineSegments[i];

			// Skip invalid segments
			if (segment.x < 0) continue;

			// line segment using Bresenham's algorithm
			DrawBresenhamLine(
				(int)segment.x, (int)segment.y,
				(int)math.round(segment.z), (int)math.round(segment.w),
				textureColors, streamlineColor, currentSize, streamlineThickness
			);
		}
	}

	private void DrawBresenhamLine(int x0, int y0, int x1, int y1, Color[] colors, Color lineColor, int size, float thickness)
	{
		bool steep = Mathf.Abs(y1 - y0) > Mathf.Abs(x1 - x0);
		if (steep)
		{
			// Swap x0, y0
			int temp = x0;
			x0 = y0;
			y0 = temp;

			// Swap x1, y1
			temp = x1;
			x1 = y1;
			y1 = temp;
		}

		if (x0 > x1)
		{
			// Swap x0, x1
			int temp = x0;
			x0 = x1;
			x1 = temp;

			// Swap y0, y1
			temp = y0;
			y0 = y1;
			y1 = temp;
		}

		int dx = x1 - x0;
		int dy = Mathf.Abs(y1 - y0);
		int error = dx / 2;

		int y = y0;
		int ystep = (y0 < y1) ? 1 : -1;
		int halfThick = (int)Mathf.Floor(thickness / 2);

		for (int x = x0; x <= x1; x++)
		{
			// Draw the point with thickness
			for (int tx = -halfThick; tx <= halfThick; tx++)
			{
				for (int ty = -halfThick; ty <= halfThick; ty++)
				{
					int drawX = steep ? y + tx : x + tx;
					int drawY = steep ? x + ty : y + ty;

					// Check bounds
					if (drawX >= 0 && drawX < size && drawY >= 0 && drawY < size)
					{
						int pixelIndex = drawX + drawY * size;
						if (pixelIndex >= 0 && pixelIndex < colors.Length)
						{
							colors[pixelIndex] = lineColor;
						}
					}
				}
			}

			error -= dy;
			if (error < 0)
			{
				y += ystep;
				error += dx;
			}
		}
	}

	[BurstCompile]
	public struct UpdateVisualizationJob : IJobParallelFor
	{
		[WriteOnly] public NativeArray<Color> colors;
		[ReadOnly] public NativeArray<float> density;
		[ReadOnly] public NativeArray<bool> obstacles;
		[ReadOnly] public NativeArray<float> pressure;

		// Visualization parameters
		public int size;
		public float sourceX;
		public float sourceY;
		public float visualMarkerRadius;
		public ColorMode colorMode;
		public Color fluidColor;
		public Color obstacleColor;
		public Color sourcePositionColor;
		public float colourIntensity;
		public bool visualizeSourcePosition;
		public bool enableCustomSource;
		public float mediumDensityThreshold;
		public float highDensityThreshold;
		public Color lowDensityColor;
		public Color mediumDensityColor;
		public Color highDensityColor;
		public bool usePressure;
		public Color lowPressureColor;
		public Color neutralPressureColor;
		public Color highPressureColor;
		public float lowPressureThreshold;
		public float highPressureThreshold;

		// Gradient data 
		[ReadOnly] public NativeArray<Color> gradientColors;
		[ReadOnly] public NativeArray<float> gradientTimes;
		public int gradientKeyCount;

		public void Execute(int index)
		{
			int i = index % size;
			int j = index / size;
			int idx = i + j * size;

			if (obstacles[idx])
			{
				// Draw obstacles with specified color
				colors[idx] = obstacleColor;
				return; // Skip regular fluid coloring for obstacle cells
			}

			float d = density[idx];
			float normalizedD = d * colourIntensity;
			Color pixelColor;

			// Apply different color modes
			switch (colorMode)
			{
				case ColorMode.DensityBased:
					// Use different colors based on density thresholds
					if (d < mediumDensityThreshold)
					{
						// Lerp between zero (black) and low density color
						float t = d / mediumDensityThreshold;
						pixelColor = Color.Lerp(Color.black, lowDensityColor, t);
					}
					else if (d < highDensityThreshold)
					{
						// Lerp between low and medium density colors
						float t = (d - mediumDensityThreshold) / (highDensityThreshold - mediumDensityThreshold);
						pixelColor = Color.Lerp(lowDensityColor, mediumDensityColor, t);
					}
					else
					{
						// Lerp between medium and high density colors
						float t = math.min(1f, (d - highDensityThreshold) / highDensityThreshold);
						pixelColor = Color.Lerp(mediumDensityColor, highDensityColor, t);
					}
					break;

				case ColorMode.Gradient:
					// Use gradient evaluation (normalizedD is clamped to 0-1 range)
					float clampedValue = math.clamp(normalizedD, 0f, 1f);
					pixelColor = EvaluateGradient(clampedValue);
					break;

				case ColorMode.SingleColor:
				default:
					// Use single colour with varying intensity
					pixelColor = new Color(
						fluidColor.r * normalizedD,
						fluidColor.g * normalizedD,
						fluidColor.b * normalizedD,
						fluidColor.a
					);
					break;

				case ColorMode.PressureBased:
					float p = pressure[idx];
					if (p < lowPressureThreshold)
					{
						float t = p / lowPressureThreshold; // Normalize to [0,1]
						pixelColor = Color.Lerp(lowPressureColor, neutralPressureColor, 1f + t);
					}
					else if (p <= highPressureThreshold)
					{
						float t = (p - lowPressureThreshold) / (highPressureThreshold - lowPressureThreshold);
						pixelColor = Color.Lerp(neutralPressureColor, highPressureColor, t);
					}
					else
					{
						float t = math.min(1f, (p - highPressureThreshold) / highPressureThreshold);
						pixelColor = Color.Lerp(highPressureColor, new Color(1f, 0.5f, 0f), t); // Transition to orange for very high pressure
					}
					break;
			}

			colors[idx] = pixelColor;

			// Visualize source position if enabled
			if (visualizeSourcePosition && enableCustomSource)
			{
				float distSq = (i - sourceX) * (i - sourceX) + (j - sourceY) * (j - sourceY);
				if (distSq < visualMarkerRadius * visualMarkerRadius)
				{
					// Mark the source position with a distinct color
					colors[idx] = sourcePositionColor;
				}
			}
		}

		private Color EvaluateGradient(float time)
		{
			if (gradientKeyCount <= 0)
				return Color.white;

			if (time <= gradientTimes[0])
				return gradientColors[0];

			if (time >= gradientTimes[gradientKeyCount - 1])
				return gradientColors[gradientKeyCount - 1];

			// Find the keys needed to interpolate between
			int index = 0;
			while (index < gradientKeyCount - 1 && time > gradientTimes[index + 1])
			{
				index++;
			}

			float t = (time - gradientTimes[index]) / (gradientTimes[index + 1] - gradientTimes[index]);
			return Color.Lerp(gradientColors[index], gradientColors[index + 1], t);
		}
	}

	public void SaveCurrentConfiguration()
	{
		SQL.SaveSimRunParams(
		size,
		diffusion,
		viscosity,
		timeStep,
		enableCustomSource,
		sourceStrength,
		sourcePositionX,
		sourcePositionY,
		enableObstacle,
		obstacleShape.ToString(),
		obstaclePositionX,
		obstaclePositionY,
		obstacleRadius,
		obstacleWidth,
		obstacleHeight
		);
	}
}