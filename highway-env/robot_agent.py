import asyncio
import websockets
import json
import torch  # Für GPU-Training (PyTorch)

async def control_vehicle():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        for _ in range(10000):  # 10 Schritte ausführen
            # Beobachtung vom Server empfangen
            obs_data = await websocket.recv()
            observation = json.loads(obs_data)
            print("Empfangene Beobachtung:", observation[:5])  # Nur ersten Werte anzeigen

            # Berechne eine zufällige Aktion (später mit ML-Modell ersetzen)
            action = torch.randint(0, 5, (1,)).item()  # Zufällige Aktion (0-4)
            print("Gesendete Aktion:", action)

            # Aktion an Server senden
            await websocket.send(json.dumps(action))

        print("Fahrzeugsteuerung abgeschlossen.")

if __name__ == "__main__":
    asyncio.run(control_vehicle())
