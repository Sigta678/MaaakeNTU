import RPi.GPIO as GPIO
import time
import threading
import paho.mqtt.client as mqtt

CONTROL_PIN = 7
PWM_FREQ = 50
STEP=0

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(CONTROL_PIN, GPIO.OUT)
 
pwm = GPIO.PWM(CONTROL_PIN, PWM_FREQ)
pwm.start(0)

def angle_to_duty_cycle(angle=0):
    duty_cycle = (0.05 * PWM_FREQ) + (0.19 * PWM_FREQ * angle / 180)
    return duty_cycle

def on_connect(client, userdata, flags, rc):
    client.subscribe("ghxPz8bRMXs8cTpvvFA7QFRRvFP6B78c/bar")

def on_message(client, userdata, msg):
    STEP = 0
    print(msg.payload)
    position = str(msg.payload.decode('ascii'))
    print(position)
    pos = position.split(',')
    pos[0] = float(pos[0])/640.0
    if(pos[0]>0 and pos[0]<0.30):
        STEP=90
    elif(pos[0]>=0.25 and pos[0]<0.43):
        STEP=67
    elif(pos[0]<=0.43 and pos[0]<0.57):
        STEP=45
    elif(pos[0]<=0.57 and pos[0]<0.75):
        STEP=22
    else:
        STEP=0
    print("move {}".format(STEP))
    dc=angle_to_duty_cycle(STEP)
    pwm.ChangeDutyCycle(dc)
    time.sleep(0.5)

def job():
    client.loop_forever()
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("broker.hivemq.com", 1883, 60)

t = threading.Thread(target = job)

t.start()

try:
    while True:
        next
except KeyboardInterrupt:
    print("closed")
finally:
    t.join()
    pwm.stop()
    GPIO.cleanup()
