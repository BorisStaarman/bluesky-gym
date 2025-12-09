"""
Test script to validate CPA risk function behavior
"""
import numpy as np
import matplotlib.pyplot as plt
from bluesky_gym.envs.ma_env_ppo import SectorEnv

def test_cpa_risk_scenarios():
    """Test CPA risk function with known scenarios"""
    print("Testing CPA Risk Function")
    print("=" * 50)
    
    # Test Case 1: Head-on collision course
    print("1. Head-on collision (high risk)")
    dx, dy = 0, 100  # 100m apart vertically
    dvx, dvy = 0, -20  # approaching at 20 m/s
    risk1 = SectorEnv._cpa_risk(dx, dy, dvx, dvy)
    print(f"   Risk: {risk1:.3f} (should be high, ~0.8-1.0)")
    
    # Test Case 2: Parallel paths (low risk)
    print("\n2. Parallel paths (low risk)")
    dx, dy = 50, 0   # 50m apart horizontally
    dvx, dvy = 0, 10  # both moving in same direction
    risk2 = SectorEnv._cpa_risk(dx, dy, dvx, dvy)
    print(f"   Risk: {risk2:.3f} (should be low, ~0.0-0.3)")
    
    # Test Case 3: Diverging (very low risk)
    print("\n3. Diverging paths (very low risk)")
    dx, dy = 50, 50   # 50m apart diagonally
    dvx, dvy = 10, 10  # moving away from each other
    risk3 = SectorEnv._cpa_risk(dx, dy, dvx, dvy)
    print(f"   Risk: {risk3:.3f} (should be very low, ~0.0-0.1)")
    
    # Test Case 4: Close but slow approach
    print("\n4. Close but slow approach")
    dx, dy = 20, 0   # 20m apart
    dvx, dvy = -1, 0  # slow approach 1 m/s
    risk4 = SectorEnv._cpa_risk(dx, dy, dvx, dvy)
    print(f"   Risk: {risk4:.3f} (should be medium, ~0.3-0.6)")
    
    # Test Case 5: Fast but distant
    print("\n5. Fast but distant")
    dx, dy = 0, 500  # 500m apart
    dvx, dvy = 0, -30  # fast approach 30 m/s
    risk5 = SectorEnv._cpa_risk(dx, dy, dvx, dvy)
    print(f"   Risk: {risk5:.3f} (should be low-medium, ~0.1-0.4)")
    
    # Test Case 6: No relative motion
    print("\n6. No relative motion")
    dx, dy = 100, 0
    dvx, dvy = 0, 0   # no relative velocity
    risk6 = SectorEnv._cpa_risk(dx, dy, dvx, dvy)
    print(f"   Risk: {risk6:.3f} (should be very low, ~0.0)")
    
    print("\n" + "=" * 50)
    print("Summary of risks:")
    print(f"Head-on:      {risk1:.3f}")
    print(f"Parallel:     {risk2:.3f}")
    print(f"Diverging:    {risk3:.3f}")
    print(f"Close-slow:   {risk4:.3f}")
    print(f"Fast-distant: {risk5:.3f}")
    print(f"No motion:    {risk6:.3f}")

def visualize_risk_heatmap():
    """Create a heatmap showing risk values for different scenarios"""
    print("\nGenerating risk heatmap...")
    
    # Create grid of relative positions
    positions = np.linspace(-200, 200, 40)
    velocities = np.linspace(-30, 30, 40)
    
    X, Y = np.meshgrid(positions, velocities)
    risks = np.zeros_like(X)
    
    # Fixed scenario: intruder approaching from the side
    fixed_dy = 0  # same altitude
    fixed_dvy = 0  # no vertical relative velocity
    
    for i in range(len(positions)):
        for j in range(len(velocities)):
            dx = X[i, j]  # relative x position
            dvx = Y[i, j]  # relative x velocity
            risks[i, j] = SectorEnv._cpa_risk(dx, fixed_dy, dvx, fixed_dvy)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(1, 2, 1)
    contour = plt.contourf(X, Y, risks, levels=20, cmap='RdYlBu_r')
    plt.colorbar(contour, label='CPA Risk')
    plt.xlabel('Relative X Position (m)')
    plt.ylabel('Relative X Velocity (m/s)')
    plt.title('CPA Risk Heatmap\n(Fixed: dy=0, dvy=0)')
    plt.grid(True, alpha=0.3)
    
    # Add some test points
    test_points_x = [0, 50, -50, 0, 100]
    test_points_vx = [-20, 0, 0, 10, -15]
    test_risks = [SectorEnv._cpa_risk(x, 0, vx, 0) for x, vx in zip(test_points_x, test_points_vx)]
    
    plt.scatter(test_points_x, test_points_vx, c='white', s=100, edgecolor='black', linewidth=2)
    for i, (x, vx, risk) in enumerate(zip(test_points_x, test_points_vx, test_risks)):
        plt.annotate(f'{risk:.2f}', (x, vx), xytext=(5, 5), textcoords='offset points', 
                    color='white', fontweight='bold', fontsize=10)
    
    # 3D surface plot
    ax = plt.subplot(1, 2, 2, projection='3d')
    surf = ax.plot_surface(X, Y, risks, cmap='RdYlBu_r', alpha=0.8)
    ax.set_xlabel('Relative X Position (m)')
    ax.set_ylabel('Relative X Velocity (m/s)')
    ax.set_zlabel('CPA Risk')
    ax.set_title('CPA Risk 3D Surface')
    
    plt.tight_layout()
    plt.show()

def test_risk_ranking():
    """Test if risk ranking works as expected"""
    print("\nTesting risk ranking...")
    
    # Create several intruder scenarios
    scenarios = [
        {"name": "Head-on close", "dx": 0, "dy": 50, "dvx": 0, "dvy": -15},
        {"name": "Side approach", "dx": 100, "dy": 0, "dvx": -10, "dvy": 0},
        {"name": "Parallel distant", "dx": 200, "dy": 0, "dvx": 0, "dvy": 5},
        {"name": "Diverging", "dx": 80, "dy": 80, "dvx": 5, "dvy": 5},
        {"name": "Very close slow", "dx": 30, "dy": 0, "dvx": -2, "dvy": 0},
    ]
    
    # Calculate risks
    risks = []
    for scenario in scenarios:
        risk = SectorEnv._cpa_risk(scenario["dx"], scenario["dy"], 
                                 scenario["dvx"], scenario["dvy"])
        risks.append((risk, scenario["name"]))
    
    # Sort by risk (descending)
    risks.sort(reverse=True)
    
    print("Intruders ranked by collision risk:")
    for i, (risk, name) in enumerate(risks, 1):
        print(f"{i}. {name:<20} Risk: {risk:.3f}")
    
    print("\nExpected order: Head-on close > Very close slow > Side approach > Diverging > Parallel distant")

if __name__ == "__main__":
    # Run all tests
    test_cpa_risk_scenarios()
    test_risk_ranking()
    
    # Ask user if they want to see visualization
    response = input("\nDo you want to see the risk heatmap visualization? (y/n): ")
    if response.lower().startswith('y'):
        visualize_risk_heatmap()