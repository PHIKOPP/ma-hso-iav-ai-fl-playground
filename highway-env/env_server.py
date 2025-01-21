import asyncio
import websockets
import gymnasium as gym
import highway_env
import json

# Highway-Umgebung mit angepasster Konfiguration erstellen
env = gym.make("highway-v0", render_mode="human")

env.unwrapped.configure({
    "observation": {
        "type": "Kinematics"
    },
    "action": {
        "type": "DiscreteMetaAction"
    },
    "lanes_count": 4,  # Anzahl der Fahrspuren
    "vehicles_count": 20,  # Anzahl der Fahrzeuge auf der Straße
    "duration": 20,  # Episodendauer in Sekunden
    "initial_spacing": 2,
    "collision_reward": -1,  # Strafe bei Kollisionen
    "reward_speed_range": [20, 30],  # Geschwindigkeitsbelohnung
    "simulation_frequency": 15,  # Simulationsrate (Hz)
    "policy_frequency": 5,  # Agent-Entscheidungen pro Sekunde
    "screen_width": 800,  # Fensterbreite
    "screen_height": 200,  # Fensterhöhe
    "centering_position": [0.3, 0.5],  # Position des Ego-Fahrzeugs auf dem Bildschirm
    "scaling": 5.5,  # Maßstab
    "render_agent": True,  # Ego-Fahrzeug anzeigen
})

env.reset()

async def handler(websocket):
    print("Client verbunden.")
    try:
        while True:
            # Beobachtung und zusätzliche Infos holen
            observation, reward, done, truncated, info = env.step(env.action_space.sample())

            # Überprüfen, ob eine Kollision passiert ist
            if done or info.get("crashed", False):
                print("Kollision erkannt oder Episode beendet! Reset wird durchgeführt.")
                env.reset()

            # Beobachtung an den Client senden
            obs_data = json.dumps(observation.tolist())
            await websocket.send(obs_data)

            # Warte auf Aktion vom Client
            action = await websocket.recv()
            action = json.loads(action)
            print("Erhaltene Aktion:", action)

            # Aktion in der Simulation ausführen
            env.step(action)

            # Umgebung rendern
            env.render()

    except websockets.exceptions.ConnectionClosed:
        print("Client getrennt.")
        env.close()

async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("Server läuft auf ws://localhost:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
