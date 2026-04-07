#!/serge/bin/venv python3
"""
Live Gesichtserkennung mit Kameraaufnahme und Video-Speicherung
- OpenCV + Haar Cascade (schnell, einfach)
- Live-Vorschau, Tastensteuerung
- Automatisches Video-Speichern
"""

import cv2
import os
from pathlib import Path

def main():
    # Haar-Cascade laden (OpenCV integriert)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Kamera öffnen (0 = Standard-Webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Kamera konnte nicht geöffnet werden!")
        return
    
    # Video-Recorder vorbereiten
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    recording = False
    video_filename = None
    
    # Fenster erstellen
    cv2.namedWindow('Gesichtserkennung Live', cv2.WINDOW_NORMAL)
    
    print("""
🔴  LIVE GESICHTSERKENUNG
=======================
r = Aufnahme starten/stoppen
q = Beenden
s = Screenshot speichern
""")
    
    frame_count = 0
    fps_counter = 0
    fps_start = cv2.getTickCount()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Kamera-Fehler!")
            break
        
        frame_count += 1
        
        # FPS berechnen (alle 30 Frames)
        if frame_count % 30 == 0:
            fps_end = cv2.getTickCount()
            fps = 30.0 / ((fps_end - fps_start) / cv2.getTickFrequency())
            fps_start = fps_end
            fps_counter = fps
        
        # Graustufen für schnellere Erkennung
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Gesichter erkennen
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        # Rechtecke um Gesichter zeichnen
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'Gesicht: {w}x{h}', 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 0), 2)
        
        # Aufnahme-Status anzeigen
        status_color = (0, 255, 0) if recording else (0, 0, 255)
        status_text = "🔴 AUFNEHMEN" if recording else "⏸️ PAUSIERT"
        cv2.rectangle(frame, (10, 10), (220, 40), status_color, -1)
        cv2.putText(frame, status_text, (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # FPS anzeigen
        cv2.putText(frame, f'FPS: {fps_counter:.1f}', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Gesichter: {len(faces)}', (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Fenster anzeigen
        cv2.imshow('Gesichtserkennung Live', frame)
        
        # Tastatursteuerung
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Aufnahme togglen
            recording = not recording
            
            if recording:
                # Video-Recorder starten
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                video_filename = f"gesichtserkennung_{timestamp}.mp4"
                out = cv2.VideoWriter(video_filename, fourcc, 30.0, 
                                    (int(cap.get(3)), int(cap.get(4))))
                print(f"📹 Aufnahme gestartet: {video_filename}")
            else:
                # Aufnahme stoppen
                if out:
                    out.release()
                    print(f"✅ Video gespeichert: {video_filename}")
                out = None
        elif key == ord('s'):
            # Screenshot
            screenshot_name = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(screenshot_name, frame)
            print(f"📸 Screenshot gespeichert: {screenshot_name}")
    
    # Aufräumen
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    print("👋 Programm beendet.")

if __name__ == "__main__":
    main()
