#!/usr/bin/env python3
"""
🧠 Live Perceptron Network Visualization Demo
==============================================

This shows the actual neural network diagram with circles (neurons) and arrows (connections)
updating in real-time as the network processes data!

You'll see:
- 🔵 Blue circles = Input neurons
- 🔴 Red circles = Hidden neurons
- 🟢 Green circles = Output neurons
- ➡️ Arrows = Connections/weights
- 💡 Brightness = Neuron activation level
- 📊 Arrow thickness = Weight strength

Usage:
    python3 live_network_demo.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.live_network_viz import LivePerceptronNetwork, create_live_network_demo
import numpy as np
import time


def simple_perceptron_demo():
    """Simple demo showing perceptron network structure with circles and arrows."""
    print("🧠" + "="*60 + "🧠")
    print("    LIVE PERCEPTRON NETWORK VISUALIZATION")
    print("🧠" + "="*60 + "🧠")
    print()
    print("📊 What you'll see:")
    print("   🔵 Blue circles = Input neurons")
    print("   🔴 Red circles = Hidden neurons")
    print("   🟢 Green circles = Output neurons")
    print("   ➡️  Arrows = Connections (weights)")
    print("   💡 Brightness = How active each neuron is")
    print("   📏 Arrow thickness = Weight strength")
    print()
    print("🎬 The network will process different data patterns")
    print("   and you'll see neurons light up in real-time!")
    print()

    input("👀 Press ENTER to start the live visualization... ")
    print()

    # Create different network architectures to demo
    demos = [
        {
            'name': 'Simple Network',
            'structure': [3, 4, 2],
            'description': '3 inputs → 4 hidden → 2 outputs'
        },
        {
            'name': 'Deep Network',
            'structure': [4, 6, 4, 3],
            'description': '4 inputs → 6 hidden → 4 hidden → 3 outputs'
        },
        {
            'name': 'Wide Network',
            'structure': [2, 8, 1],
            'description': '2 inputs → 8 hidden → 1 output'
        }
    ]

    for i, demo in enumerate(demos, 1):
        print(f"🎯 Demo {i}/{len(demos)}: {demo['name']}")
        print(f"   Structure: {demo['description']}")
        print(f"   Network: {' → '.join(map(str, demo['structure']))}")
        print()

        # Create network
        network = LivePerceptronNetwork(demo['structure'])
        network.start_visualization()

        print("🎬 Starting animation - watch the circles and arrows!")
        print("   (Network will process data for 30 seconds)")
        print()

        try:
            # Run animation for 30 seconds
            start_time = time.time()
            iteration = 0

            while time.time() - start_time < 30:
                iteration += 1

                # Create different data patterns
                if iteration % 20 == 0:
                    # Pattern 1: All ones
                    input_data = [1.0] * demo['structure'][0]
                    print(f"📊 Processing pattern: ALL ONES {input_data}")
                elif iteration % 20 == 10:
                    # Pattern 2: Alternating
                    input_data = [1.0 if i % 2 == 0 else 0.0 for i in range(demo['structure'][0])]
                    print(f"📊 Processing pattern: ALTERNATING {input_data}")
                elif iteration % 15 == 5:
                    # Pattern 3: Single spike
                    input_data = [0.0] * demo['structure'][0]
                    input_data[0] = 2.0
                    print(f"📊 Processing pattern: SINGLE SPIKE {input_data}")
                else:
                    # Random data
                    input_data = np.random.randn(demo['structure'][0])
                    if iteration % 10 == 0:
                        print(f"📊 Processing pattern: RANDOM {[f'{x:.2f}' for x in input_data]}")

                # Animate the network
                network.animate_data_flow(input_data)
                time.sleep(0.3)

        except KeyboardInterrupt:
            print("\n⏹️ Demo stopped by user")

        network.stop_visualization()

        if i < len(demos):
            print(f"✅ {demo['name']} demo completed!")
            print()
            input("Press ENTER to continue to next demo... ")
            print()

    print("🎉" + "="*50 + "🎉")
    print("    ALL DEMOS COMPLETED!")
    print("🎉" + "="*50 + "🎉")
    print()
    print("🧠 What you just saw:")
    print("   • Neurons (circles) changing brightness based on activation")
    print("   • Connections (arrows) showing weight strength and direction")
    print("   • Data flowing through the network in real-time")
    print("   • Different network architectures processing information")
    print()
    print("💡 Understanding the visualization:")
    print("   • Brighter neurons = More activated")
    print("   • Thicker arrows = Stronger connections")
    print("   • Blue arrows = Positive weights")
    print("   • Red arrows = Negative weights")
    print()


def interactive_network_demo():
    """Interactive demo where user can input data."""
    print("🎮" + "="*50 + "🎮")
    print("    INTERACTIVE NETWORK DEMO")
    print("🎮" + "="*50 + "🎮")
    print()
    print("🎯 You control the input data!")
    print("   Enter values and watch how the network responds")
    print()

    # Simple 3-input network
    network = LivePerceptronNetwork([3, 4, 2])
    network.start_visualization()

    print("🧠 Network created: 3 inputs → 4 hidden → 2 outputs")
    print()
    print("📝 Instructions:")
    print("   - Enter 3 numbers separated by spaces (e.g., '1 0 -1')")
    print("   - Type 'quit' to exit")
    print("   - Watch how different inputs affect the network!")
    print()

    try:
        while True:
            user_input = input("🔢 Enter 3 input values (or 'quit'): ").strip()

            if user_input.lower() == 'quit':
                break

            try:
                # Parse user input
                values = list(map(float, user_input.split()))

                if len(values) != 3:
                    print("❌ Please enter exactly 3 numbers!")
                    continue

                print(f"📊 Processing your input: {values}")

                # Animate the network with user data
                for _ in range(10):  # Show animation for 3 seconds
                    network.animate_data_flow(values)
                    time.sleep(0.3)

                print("✅ Animation complete! Try different values.")
                print()

            except ValueError:
                print("❌ Please enter valid numbers!")
                continue

    except KeyboardInterrupt:
        print("\n⏹️ Interactive demo stopped")

    network.stop_visualization()
    print("👋 Thanks for trying the interactive demo!")


def main():
    """Main function."""
    print("🧠 Live Perceptron Network Visualization")
    print("=" * 50)
    print()
    print("Choose a demo:")
    print("1. 🎬 Automatic demo (recommended for first time)")
    print("2. 🎮 Interactive demo (you control the inputs)")
    print("3. 🚀 Quick network structure demo")
    print()

    try:
        choice = input("Enter your choice (1, 2, or 3): ").strip()

        if choice == '1':
            simple_perceptron_demo()
        elif choice == '2':
            interactive_network_demo()
        elif choice == '3':
            create_live_network_demo()
        else:
            print("❌ Invalid choice. Running automatic demo...")
            simple_perceptron_demo()

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")


if __name__ == '__main__':
    main()