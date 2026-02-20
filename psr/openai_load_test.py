import io
import itertools
import logging
import os
import random
from time import strftime

import gevent
import soundfile as sf

from locust import HttpUser, task, events
from locust.runners import MasterRunner
from atomicx import AtomicInt

def get_audio_duration(data: bytes) -> float:
    try:
        f = sf.SoundFile(io.BytesIO(data))
        duration = f.frames / f.samplerate
        logging.info(f"Audio duration: {duration:.2f}s")
        return duration
    except Exception:
        return 0.0


TEXT_SAMPLES = [
    {"lang": "en-us", "text": "Andy is from Canada, but with his long blond hair, incredibly mellow demeanor, and wardrobe of faded T-shirts, shorts, and sandals, you would think he grew up surfing on the coast of Malibu. Having worked with Endel Tulving as an undergraduate, Andy reasoned that episodic memory gives us access to a tangible memory from a specific place and time, leading us to feel confident that we are reliving a genuine moment from the past."},
    {"lang": "en-us", "text": "In contrast, familiarity can be strong, such as the sense of certainty that you have seen something or someone before, or it can be weak, such as a hunch or educated guess; either way, it doesn't give us anything specific to hold on to."},
    {"lang": "en-us", "text": "Andy was presenting his research, and I might have been a bit too direct when I voiced my skepticism that familiarity was anything more than weak episodic memory."},
    {"lang": "en-us", "text": "I anticipated some blowback, but instead of getting defensive, Andy completely disarmed me by responding, 'You should be skeptical!' After an afternoon of spirited debate, we decided to team up on an experiment to test the idea that we could identify a brain area responsible for the sense of familiarity. To sweeten the pot, we decided to make a 'beer bet'—if his prediction turned out to be right, I'd buy him a beer and vice versa."},
    {"lang": "en-us", "text": "For instance, if a subject imagined what it would be like to stuff an armadillo in a shoebox, that would create a distinctive episodic memory."},
    {"lang": "en-us", "text": "(At this point, I've lost so many beer bets to Andy, it will take years to pay off my tab.)"},
    {"lang": "en-us", "text": "We didn't yet have a working MRI scanner in the lab at Berkeley, but Mark finessed a space for us at a clinical MRI facility in the Martinez VA Medical Center, about halfway between Berkeley and Davis. The Martinez scanner was set up for routine clinical scans and didn't have state-of-the-art performance specs, so I had to MacGyver it, tweaking our procedures every way I could think of to turn this Ford Pinto into a Ferrari."},
    {"lang": "en-us", "text": "Can it fit in a shoebox?)."},
    {"lang": "en-us", "text": "Sure enough, activity in the hippocampus spiked when people saw a word and formed a memory that later helped them recall something about 109the context."},
    {"lang": "en-us", "text": "This argument was based on the fact that animals with damage to the hippocampus, as well as the patients with developmental amnesia such as those studied by Faraneh Vargha-Khadem, seemed to do fine on 'recognition memory' tests that required them to tell the difference between objects they had seen before and ones that were new."},
    {"lang": "en-us", "text": "Rats, like human infants, tend to be more interested in exploring things that are new to them than in things they have seen before."},
    {"lang": "ro",    "text": "Andy vine din Canada, dar cu părul său lung și blond, atitudinea sa extrem de calmă și garderoba de tricouri decolorate, pantaloni scurți și sandale, ai crede că a crescut făcând surf pe coasta din Malibu. După ce a lucrat cu Endel Tulving ca student, Andy a dedus că memoria episodică ne oferă acces la o amintire concretă dintr-un loc și moment specific."},
    {"lang": "fr-fr", "text": "Andy vient du Canada, mais avec ses longs cheveux blonds, son attitude incroyablement détendue et sa garde-robe de t-shirts délavés, de shorts et de sandales, on croirait qu'il a grandi en surfant sur la côte de Malibu. Ayant travaillé avec Endel Tulving en tant qu'étudiant, Andy a déduit que la mémoire épisodique nous donne accès à un souvenir tangible d'un lieu et d'un moment précis."},
    {"lang": "en-us", "text": "The weather today is surprisingly mild for this time of year."},
    {"lang": "en-us", "text": "She opened the door and stepped into the cold morning air."},
    {"lang": "en-us", "text": "Memory is not a recording device; it is a reconstruction."},
    {"lang": "en-us", "text": "He checked his watch, realized he was late, and started running."},
    {"lang": "en-us", "text": "The experiment failed, but the failure taught us more than success would have."},
    {"lang": "en-us", "text": "Scientists have long debated whether dreams serve any cognitive function, with some arguing they help consolidate memories and others suggesting they are simply a byproduct of neural activity."},
    {"lang": "en-us", "text": "Turn left at the traffic light, then continue straight for about two miles until you reach the old stone bridge, cross it, and the farmhouse will be on your right behind the oak trees."},
    {"lang": "en-gb", "text": "I'd rather have a cup of tea and think it over before making any decisions."},
    {"lang": "en-gb", "text": "The train to Edinburgh departs at half past seven from platform nine."},
    {"lang": "en-gb", "text": "It was a perfectly ordinary Tuesday until the package arrived, and with it, a letter that would change everything she thought she knew about her family's past."},
    {"lang": "ro",    "text": "Vremea de astăzi este surprinzător de blândă pentru această perioadă a anului."},
    {"lang": "ro",    "text": "A deschis ușa și a pășit în aerul rece al dimineții."},
    {"lang": "ro",    "text": "Memoria nu este un dispozitiv de înregistrare; este o reconstrucție activă, influențată de emoții, context și de tot ceea ce am trăit între momentul evenimentului și cel al amintirii."},
    {"lang": "ro",    "text": "Experimentul a eșuat, dar eșecul ne-a învățat mai mult decât ar fi făcut-o succesul."},
    {"lang": "ro",    "text": "Oamenii de știință dezbat de mult timp dacă visele au vreo funcție cognitivă."},
    {"lang": "fr-fr", "text": "Le temps est étonnamment doux pour cette période de l'année."},
    {"lang": "fr-fr", "text": "Elle a ouvert la porte et est entrée dans l'air froid du matin."},
    {"lang": "fr-fr", "text": "La mémoire n'est pas un simple enregistreur; c'est une reconstruction active, façonnée par nos émotions, notre contexte et tout ce que nous avons vécu depuis l'événement original."},
    {"lang": "fr-fr", "text": "L'expérience a échoué, mais l'échec nous a appris plus que le succès ne l'aurait fait."},
    {"lang": "fr-fr", "text": "Tournez à gauche au feu, puis continuez tout droit pendant environ trois kilomètres."},
]

VOICES = [
    {"id": "0(50)+1(50)", "lang": "en-us", "gender": "neutral", "name": "Voice mix"},
    {"id": "0", "lang": "en-us", "gender": "male", "name": "Kai"},
    {"id": "1", "lang": "en-us", "gender": "female", "name": "Jane"},
    {"id": "2", "lang": "en-us", "gender": "female", "name": "Aria"},
    {"id": "3", "lang": "en-gb", "gender": "female", "name": "Bella"},
    {"id": "4", "lang": "en-gb", "gender": "male", "name": "Brian"},
    {"id": "5", "lang": "en-gb", "gender": "male", "name": "Arthur"},
    {"id": "6", "lang": "en-us", "gender": "female", "name": "Nicole"},
    {"id": "7", "lang": "ro", "gender": "male", "name": "Emil"},
    {"id": "8", "lang": "fr-fr", "gender": "female", "name": "Denise"},
    {"id": "9", "lang": "fr-fr", "gender": "male", "name": "Henri"},
]

HOST = os.environ.get("HOST", "http://localhost:8880")
URL_PATH = os.environ.get("URL_PATH", "/v1/audio/speech")
OUTPUT_FORMAT = os.environ.get("KOKORO_OUTPUT_FORMAT", "ogg")

user_id = itertools.count()
concurrency = AtomicInt()

@events.test_start.add_listener
def on_test_start(environment):
    if isinstance(environment.runner, MasterRunner):
        logging.info("A new test is starting.")

@events.test_stop.add_listener
def on_test_stop(environment):
    if isinstance(environment.runner, MasterRunner):
        logging.info("The test stopped.")

class MatchaTTSClient(HttpUser):
    def __init__(self, environment):
        super().__init__(environment)
        self.id = next(user_id)
        logging.info(f"User {self.id} initialized.")

    @task
    def call_rest(self):
        current_concurrency = concurrency.inc()
        duration = 1
        try:
            sample = random.choice(TEXT_SAMPLES)
            text = sample.get("text")
            lang = sample.get("lang")
            voice = random.choice([v for v in VOICES if v["lang"] == lang])

            logging.info(f"{strftime('%X')} User {self.id} is executing, concurrency: {current_concurrency + 1}.")

            headers = {
                "Content-Type": "application/json",
                "Cache-Control": "no-store",
            }

            body = {
                "input": text,
                "voice": voice["id"],
                "response_format": OUTPUT_FORMAT,
                "stream": True,
                "speed": 1
            }

            response = self.client.post(URL_PATH, headers=headers, json=body, timeout=4)
            duration = get_audio_duration(response.content)
        finally:
            events.request.fire(
                request_type="Concurrency",
                name="Concurrency metric",
                response_time=concurrency.dec(),
                response_length=0)
        gevent.sleep(duration)
