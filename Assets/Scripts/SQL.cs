using System.Collections;
using System.Collections.Generic;
using System.Data;
using UnityEngine;
using Mono.Data.Sqlite;

using static FluidSimulation;
using System.Data.Common;
using System.Threading.Tasks;
using System;
using System.IO;
using static Unity.Burst.Intrinsics.X86.Avx;

public class SQL : MonoBehaviour
{
	//private static void EnsureTablesExist()
	//{
	//	using (var conn = new SqliteConnection("URI=file:C:\\Users\\chris\\My project (2)\\test.db"))
	//	{
	//		conn.Open();
	//		using (var cmd = conn.CreateCommand())
	//		{
	//			// Enable foreign key support
	//			cmd.CommandText = "PRAGMA foreign_keys = ON;";
	//			cmd.ExecuteNonQuery();

	//			// Modified SimulationRuns table with auto-incrementing primary key
	//			cmd.CommandText = @"
	//               CREATE TABLE IF NOT EXISTS SimulationRuns (
	//                   RunID INTEGER PRIMARY KEY AUTOINCREMENT,
	//                   Size INTEGER,
	//                   Diffusion REAL,
	//                   Viscosity REAL,
	//                   -- ... other existing columns ...
	//                   Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
	//               );";
	//			cmd.ExecuteNonQuery();

	//			// Modified RuntimeMetrics table with foreign key
	//			cmd.CommandText = @"
	//               CREATE TABLE IF NOT EXISTS RuntimeMetrics (
	//                   MetricID INTEGER PRIMARY KEY AUTOINCREMENT,
	//                   RunID INTEGER,
	//                   Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
	//                   AverageDensity REAL,
	//                   MaxVelocityMagnitude REAL,
	//                   FOREIGN KEY(RunID) REFERENCES SimulationRuns(RunID) ON DELETE CASCADE
	//               );";
	//			cmd.ExecuteNonQuery();
	//		}
	//		conn.Close();
	//	}
	//}


	public static int SaveSimRunParams(int size, float diffusion, float viscosity, float timeStep,
	   bool sourceEnabled, float sourceStrength, float sourceX, float sourceY,
	   bool obstacleEnabled, string obstacleType, float obstacleX, float obstacleY,
	   float obstacleRadius, float obstacleWidth, float obstacleHeight)
	{
		int runId = -1;

		if (Mathf.Abs(timeStep - 0.1f) < 1e-6f)
		{
			return runId; // Skip invalid timeStep
		}

		using (var conn = new SqliteConnection("URI=file:C:\\Users\\chris\\My project (2)\\test.db"))
		{
			conn.Open();
			using (var cmd = conn.CreateCommand())
			{
				cmd.CommandText = @"INSERT INTO SimulationRuns 
                    (Size, Diffusion, Viscosity, TimeStep, SourceEnabled, SourceStrength, SourcePositionX, SourcePositionY, 
                     ObstacleEnabled, ObstacleType, ObstaclePositionX, ObstaclePositionY, ObstacleRadius, ObstacleWidth, ObstacleHeight)
                    VALUES 
                    (@Size, @Diffusion, @Viscosity, @TimeStep, @SourceEnabled, @SourceStrength, @SourcePositionX, @SourcePositionY, 
                     @ObstacleEnabled, @ObstacleType, @ObstaclePositionX, @ObstaclePositionY, @ObstacleRadius, @ObstacleWidth, @ObstacleHeight)";

				cmd.Parameters.Add(new SqliteParameter("@Size", size));
				cmd.Parameters.Add(new SqliteParameter("@Diffusion", diffusion));
				cmd.Parameters.Add(new SqliteParameter("@Viscosity", viscosity));
				cmd.Parameters.Add(new SqliteParameter("@TimeStep", timeStep));
				cmd.Parameters.Add(new SqliteParameter("@SourceEnabled", sourceEnabled ? 1 : 0));
				cmd.Parameters.Add(new SqliteParameter("@SourceStrength", sourceStrength));
				cmd.Parameters.Add(new SqliteParameter("@SourcePositionX", sourceX));
				cmd.Parameters.Add(new SqliteParameter("@SourcePositionY", sourceY));
				cmd.Parameters.Add(new SqliteParameter("@ObstacleEnabled", obstacleEnabled ? 1 : 0));
				cmd.Parameters.Add(new SqliteParameter("@ObstacleType", obstacleType));
				cmd.Parameters.Add(new SqliteParameter("@ObstaclePositionX", obstacleX));
				cmd.Parameters.Add(new SqliteParameter("@ObstaclePositionY", obstacleY));
				cmd.Parameters.Add(new SqliteParameter("@ObstacleRadius", obstacleRadius));
				cmd.Parameters.Add(new SqliteParameter("@ObstacleWidth", obstacleWidth));
				cmd.Parameters.Add(new SqliteParameter("@ObstacleHeight", obstacleHeight));

				cmd.ExecuteNonQuery();
				cmd.CommandText = "SELECT last_insert_rowid()";
				runId = Convert.ToInt32(cmd.ExecuteScalar());
			}
			conn.Close();
		}
		return runId;

	}

	public static void LogRuntimeMetrics(
		int runId,
		int step,
		float avgDensity,
		float maxVelocity,
		float frameRate)
	{
		using (var conn = new SqliteConnection("URI=file:C:\\Users\\chris\\My project (2)\\test.db"))
		{
			conn.Open();
			using (var cmd = conn.CreateCommand())
			{
				cmd.CommandText = @"
                INSERT INTO RuntimeMetrics 
                (RunID, AverageDensity, MaxVelocityMagnitude, FrameRate)
                VALUES 
                (@runId, @avgDensity, @maxVelocity, @frameRate)";

				cmd.Parameters.AddRange(new[] {
				new SqliteParameter("@runId", runId),
				new SqliteParameter("@avgDensity", avgDensity),
				new SqliteParameter("@maxVelocity", maxVelocity),
				new SqliteParameter("@frameRate", frameRate)
			});

				cmd.ExecuteNonQuery();
			}
			conn.Close();
		}
	}
}

