import json
import os
import re
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from urllib.parse import urlparse

import requests

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DSP_CACHEDIR", os.path.join(RAIZ, ".cache", "dsp"))
os.environ.setdefault("HF_HOME", os.path.join(RAIZ, ".cache", "huggingface"))

from dsp.modules.gpt3 import chat_request
from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
from knowledge_storm.lm import OpenAIModel
from knowledge_storm.rm import SerperRM
from knowledge_storm.utils import ArticleTextProcessing, load_api_key

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agregar_referencias import procesar

GRUPOS_DOMINIOS = [
    [
        "pubmed.ncbi.nlm.nih.gov",
        "pmc.ncbi.nlm.nih.gov",
        "europepmc.org",
        "cochranelibrary.com",
        "apa.org",
        "psychiatryonline.org",
        "scielo.org",
        "redalyc.org",
        "dialnet.unirioja.es",
    ],
    [
        "sciencedirect.com",
        "springer.com",
        "biomedcentral.com",
        "wiley.com",
        "tandfonline.com",
        "sagepub.com",
        "frontiersin.org",
        "bmj.com",
        "cambridge.org",
    ],
    [
        "who.int",
        "unodc.org",
        "samhsa.gov",
        "nida.nih.gov",
        "nice.org.uk",
        "euda.europa.eu",
        "paho.org",
        "cicad.oas.org",
        "ncbi.nlm.nih.gov/books",
    ],
]
DOMINIOS_EXTRA_ACEPTADOS = ["ncbi.nlm.nih.gov", "nature.com", "plos.org", "jamanetwork.com", "thelancet.com", "academic.oup.com", "emcdda.europa.eu", "nih.gov"]
GRUPOS_EDUCACION = [
    [
        "eric.ed.gov",
        "dl.acm.org",
        "ieeexplore.ieee.org",
        "sciencedirect.com",
        "springer.com",
        "tandfonline.com",
        "wiley.com",
        "sagepub.com",
        "frontiersin.org",
    ],
    [
        "irrodl.org",
        "educause.edu",
        "emerald.com",
        "jstor.org",
        "apa.org",
        "pmc.ncbi.nlm.nih.gov",
        "scielo.org",
        "redalyc.org",
        "dialnet.unirioja.es",
    ],
    [
        "w3.org",
        "iso.org",
        "cast.org",
        "unesco.org",
        "oecd.org",
        "etsi.org",
        "section508.gov",
        "europa.eu",
        "jisc.ac.uk",
    ],
]
EXTRA_EDUCACION = ["ncbi.nlm.nih.gov", "nature.com", "plos.org", "acm.org", "ieee.org", "cambridge.org", "bmj.com", "biomedcentral.com", "who.int"]
GRUPOS_VINCULOS = [
    [
        "pubmed.ncbi.nlm.nih.gov",
        "pmc.ncbi.nlm.nih.gov",
        "europepmc.org",
        "cochranelibrary.com",
        "apa.org",
        "eric.ed.gov",
        "scielo.org",
        "redalyc.org",
        "dialnet.unirioja.es",
    ],
    [
        "wiley.com",
        "tandfonline.com",
        "sagepub.com",
        "springer.com",
        "sciencedirect.com",
        "cambridge.org",
        "academic.oup.com",
        "frontiersin.org",
        "biomedcentral.com",
    ],
    [
        "who.int",
        "unicef.org",
        "ohchr.org",
        "unwomen.org",
        "coe.int",
        "paho.org",
        "cdc.gov",
        "counseling.org",
        "aamft.org",
    ],
]
EXTRA_VINCULOS = ["ncbi.nlm.nih.gov", "nature.com", "plos.org", "jamanetwork.com", "thelancet.com", "bmj.com", "un.org", "oecd.org", "nice.org.uk"]
GRUPOS_ORGANIZACIONAL = [
    [
        "pubmed.ncbi.nlm.nih.gov",
        "pmc.ncbi.nlm.nih.gov",
        "cochranelibrary.com",
        "apa.org",
        "annualreviews.org",
        "eric.ed.gov",
        "scielo.org",
        "redalyc.org",
        "dialnet.unirioja.es",
    ],
    [
        "wiley.com",
        "tandfonline.com",
        "sagepub.com",
        "springer.com",
        "sciencedirect.com",
        "emerald.com",
        "journals.aom.org",
        "cambridge.org",
        "frontiersin.org",
    ],
    [
        "ilo.org",
        "who.int",
        "osha.europa.eu",
        "cdc.gov",
        "iso.org",
        "nice.org.uk",
        "insst.es",
        "counseling.org",
        "bacp.co.uk",
    ],
]
EXTRA_ORGANIZACIONAL = ["ncbi.nlm.nih.gov", "europepmc.org", "academic.oup.com", "biomedcentral.com", "bmj.com", "nature.com", "plos.org", "eurofound.europa.eu", "oecd.org", "paho.org", "journals.copmadrid.org", "un.org"]
EXCLUIDOS_ORGANIZACIONAL = r"oshwiki\.|/oshnews/|voices\.ilo\.org|/infostories/|ct\.counseling\.org|/vistas/|manifold\.counseling\.org|/bacp-journals/|apa\.org/monitor/|/webinars?/|/courses?/|/training/"
GRUPOS_ESCOLAR = [
    [
        "eric.ed.gov",
        "pubmed.ncbi.nlm.nih.gov",
        "pmc.ncbi.nlm.nih.gov",
        "cochranelibrary.com",
        "apa.org",
        "scielo.org",
        "redalyc.org",
        "dialnet.unirioja.es",
        "revistas.uned.es",
    ],
    [
        "sagepub.com",
        "wiley.com",
        "tandfonline.com",
        "springer.com",
        "sciencedirect.com",
        "cambridge.org",
        "academic.oup.com",
        "frontiersin.org",
        "journals.copmadrid.org",
    ],
    [
        "unesco.org",
        "unicef.org",
        "who.int",
        "oecd.org",
        "ohchr.org",
        "cedefop.europa.eu",
        "schoolcounselor.org",
        "iaevg.com",
        "casel.org",
    ],
]
EXTRA_ESCOLAR = ["counseling.org", "ncbi.nlm.nih.gov", "europepmc.org", "biomedcentral.com", "nature.com", "plos.org", "jstor.org", "emerald.com", "nasponline.org", "europa.eu", "ilo.org", "un.org", "cdc.gov", "nice.org.uk", "paho.org"]
EXCLUIDOS_ESCOLAR = r"ct\.counseling\.org|/vistas/|manifold\.counseling\.org|apa\.org/monitor/|school-counselor-magazine|/magazines?/|/newsletters?/|unesco\.org/[a-z]{2}/articles/|/webinars?/|/courses?/|/training/|/conferences?/|/awards?/|/advocacy/"
GRUPOS_ADICCIONES = [
    [
        "pubmed.ncbi.nlm.nih.gov",
        "pmc.ncbi.nlm.nih.gov",
        "europepmc.org",
        "cochranelibrary.com",
        "apa.org",
        "psychiatryonline.org",
        "scielo.org",
        "redalyc.org",
        "dialnet.unirioja.es",
    ],
    [
        "sciencedirect.com",
        "wiley.com",
        "tandfonline.com",
        "sagepub.com",
        "springer.com",
        "biomedcentral.com",
        "jamanetwork.com",
        "thelancet.com",
        "academic.oup.com",
    ],
    [
        "who.int",
        "unodc.org",
        "samhsa.gov",
        "nida.nih.gov",
        "niaaa.nih.gov",
        "nice.org.uk",
        "euda.europa.eu",
        "paho.org",
        "healthquality.va.gov",
    ],
]
EXTRA_ADICCIONES = ["ncbi.nlm.nih.gov", "nih.gov", "bmj.com", "cambridge.org", "frontiersin.org", "nature.com", "plos.org", "annualreviews.org", "jsad.com", "journals.lww.com", "adicciones.es", "uspreventiveservicestaskforce.org", "asam.org", "cicad.oas.org", "emcdda.europa.eu"]
EXCLUIDOS_ADICCIONES = r"/drugfacts/|/research-topics/|/brochures-and-fact-sheets/|/news-events/|/find-help/|/recovery-month/|/kids?/|/teens?/|/parents?/|apa\.org/monitor/|/webinars?/|/courses?/|/training/|/conferences?/|/awards?/|/advocacy/"
GRUPOS_PSICOTERAPIA = [
    [
        "pubmed.ncbi.nlm.nih.gov",
        "pmc.ncbi.nlm.nih.gov",
        "europepmc.org",
        "cochranelibrary.com",
        "apa.org",
        "psychiatryonline.org",
        "annualreviews.org",
        "scielo.org",
        "dialnet.unirioja.es",
    ],
    [
        "sciencedirect.com",
        "wiley.com",
        "tandfonline.com",
        "springer.com",
        "cambridge.org",
        "sagepub.com",
        "jamanetwork.com",
        "jmir.org",
        "academic.oup.com",
    ],
    [
        "who.int",
        "nice.org.uk",
        "efpa.eu",
        "iupsys.net",
        "cop.es",
        "colpsic.org.co",
        "iso.org",
        "eur-lex.europa.eu",
        "asppb.net",
    ],
]
EXTRA_PSICOTERAPIA = ["ncbi.nlm.nih.gov", "nih.gov", "bmj.com", "thelancet.com", "frontiersin.org", "nature.com", "plos.org", "biomedcentral.com", "redalyc.org", "papelesdelpsicologo.es", "journals.copmadrid.org", "revistas.uned.es", "guilfordjournals.com", "liebertpub.com", "paho.org", "wpanet.org", "div12.org", "europa.eu"]
EXCLUIDOS_PSICOTERAPIA = r"apa\.org/monitor/|apa\.org/education-career/|/helpcenter/|/find-a-|/self-help/|preprints\.jmir\.org|/webinars?/|/courses?/|/training/|/continuing-education/|/conferences?/|/awards?/|/advocacy/"
GRUPOS_ADICCIONES_INTEGRAL = [
    [
        "pubmed.ncbi.nlm.nih.gov",
        "pmc.ncbi.nlm.nih.gov",
        "europepmc.org",
        "cochranelibrary.com",
        "apa.org",
        "psychiatryonline.org",
        "scielo.org",
        "redalyc.org",
        "dialnet.unirioja.es",
    ],
    [
        "sciencedirect.com",
        "wiley.com",
        "tandfonline.com",
        "sagepub.com",
        "springer.com",
        "jamanetwork.com",
        "thelancet.com",
        "nature.com",
        "academic.oup.com",
    ],
    [
        "biomedcentral.com",
        "frontiersin.org",
        "cambridge.org",
        "bmj.com",
        "karger.com",
        "journals.lww.com",
        "annualreviews.org",
        "jsad.com",
        "adicciones.es",
    ],
    [
        "who.int",
        "unodc.org",
        "samhsa.gov",
        "nida.nih.gov",
        "niaaa.nih.gov",
        "nice.org.uk",
        "euda.europa.eu",
        "paho.org",
        "healthquality.va.gov",
    ],
    [
        "incb.org",
        "un.org",
        "cicad.oas.org",
        "eur-lex.europa.eu",
        "ema.europa.eu",
        "ohchr.org",
        "coe.int",
        "fda.gov",
        "uspreventiveservicestaskforce.org",
    ],
]
EXTRA_ADICCIONES_INTEGRAL = ["ncbi.nlm.nih.gov", "nih.gov", "asam.org", "cdc.gov", "plos.org", "jstor.org", "europa.eu", "emcdda.europa.eu", "iris.who.int", "hhs.gov", "ahrq.gov", "journals.copmadrid.org", "revistas.uned.es", "wpanet.org"]
EXCLUIDOS_ADICCIONES_INTEGRAL = r"/drugfacts/|/research-topics/|/brochures-and-fact-sheets/|/news-events/|/find-help/|/recovery-month/|/kids?/|/teens?/|/parents?/|/consumers?/|/consumer-updates/|/patients?/|/campaigns?/|/stopoverdose/|/rxawareness/|/overdose-resources/|annualconference\.asam\.org|elearning\.asam\.org|go\.nature\.com|/tracking/|assetclicktracking|sk\.sagepub\.com|/sitefinity-production-blobs/|accessdata\.fda\.gov|stacks\.cdc\.gov|ftp\.cdc\.gov|/infographics?/|/quizzes?/|europepmc\.org/article/ppr/|/preprints?/|apa\.org/monitor/|/webinars?/|/courses?/|/training/|/conferences?/|/awards?/|/advocacy/"
GRUPOS_MARCA_PROFESIONAL = [
    [
        "pubmed.ncbi.nlm.nih.gov",
        "pmc.ncbi.nlm.nih.gov",
        "europepmc.org",
        "apa.org",
        "eric.ed.gov",
        "jstor.org",
        "scielo.org",
        "redalyc.org",
        "dialnet.unirioja.es",
    ],
    [
        "wiley.com",
        "tandfonline.com",
        "sagepub.com",
        "springer.com",
        "tpcjournal.nbcc.org",
        "cambridge.org",
        "academic.oup.com",
        "brill.com",
        "meridian.allenpress.com",
    ],
    [
        "sciencedirect.com",
        "emerald.com",
        "frontiersin.org",
        "jmir.org",
        "liebertpub.com",
        "journalofethics.ama-assn.org",
        "jamanetwork.com",
        "bmj.com",
        "journals.lww.com",
    ],
    [
        "counseling.org",
        "nbcc.org",
        "bacp.co.uk",
        "ccpa-accp.ca",
        "cop.es",
        "colpsic.org.co",
        "ftc.gov",
        "asa.org.uk",
        "eur-lex.europa.eu",
    ],
]
EXTRA_MARCA_PROFESIONAL = ["ncbi.nlm.nih.gov", "nih.gov", "biomedcentral.com", "plos.org", "nature.com", "annualreviews.org", "cacrep.org", "aamft.org", "iac-irtac.org", "efpa.eu", "iupsys.net", "asppb.net", "psychotherapy.org.uk", "pacfa.org.au", "hhs.gov", "ecfr.gov", "federalregister.gov", "govinfo.gov", "ico.org.uk", "europa.eu", "papelesdelpsicologo.es", "journals.copmadrid.org", "revistas.uned.es", "guilfordjournals.com"]
EXCLUIDOS_MARCA_PROFESIONAL = r"ct\.counseling\.org|/counseling-today-magazine/|/tag/|/category/|/vistas/|manifold\.counseling\.org|/bacp-journals/|/therapy-today/|apa\.org/monitor/|apa\.org/education-career/|/helpcenter/|/find-a-|/self-help/|consumer\.ftc\.gov|/consumer-advice/|/news-events/|sk\.sagepub\.com|eric\.ed\.gov/\?id=ED|preprints\.jmir\.org|europepmc\.org/article/ppr/|/preprints?/|/webinars?/|/courses?/|/training/|/continuing-education/|/conferences?/|/awards?/|/advocacy/|/marketplace/|/directory/"
PERSONAS_EXTRA = {
    "marca_profesional": [
        "Counseling ethicist and licensure board consultant: Covers the advertising, public-statement and social-media standards of the ACA Code of Ethics (2014), NBCC, BACP, CCPA and APA Standard 5, including confidentiality in promotion, solicitation of testimonials, use of case examples and client consent, claims about outcomes and credentials, and avoiding confusion about the scope of counseling versus psychotherapy, psychology or coaching.",
        "Scholar of professional identity and reputation in the helping professions: Covers counselor professional identity development, personal branding and reputation research, value propositions and differentiation within scope of competence, online professionalism and digital presence, and consumer-protection rules on endorsements and health claims (FTC, ASA/CAP, GDPR).",
    ],
    "adicciones_integral": [
        "Addiction pharmacologist focused on stimulants, cannabis, hallucinogens and dissociatives: Covers the neurochemistry (dopamine, serotonin, glutamate and endocannabinoid systems) and clinical effects of cocaine, amphetamines and methamphetamine, cannabis, classic psychedelics, ketamine and phencyclidine, and the peer-reviewed evidence on their treatment and management.",
        "Public health and drug policy specialist: Covers international drug control conventions (1961, 1971 and 1988), the work of the WHO, UNODC and the INCB, regulation of medicines used in addiction treatment, harm reduction policy, and continuity of care across health systems.",
    ],
}
PERFILES_FUENTES = {
    "salud": (GRUPOS_DOMINIOS, DOMINIOS_EXTRA_ACEPTADOS),
    "educacion": (GRUPOS_EDUCACION, EXTRA_EDUCACION),
    "vinculos": (GRUPOS_VINCULOS, EXTRA_VINCULOS),
    "organizacional": (GRUPOS_ORGANIZACIONAL, EXTRA_ORGANIZACIONAL, EXCLUIDOS_ORGANIZACIONAL),
    "escolar": (GRUPOS_ESCOLAR, EXTRA_ESCOLAR, EXCLUIDOS_ESCOLAR),
    "adicciones": (GRUPOS_ADICCIONES, EXTRA_ADICCIONES, EXCLUIDOS_ADICCIONES),
    "psicoterapia": (GRUPOS_PSICOTERAPIA, EXTRA_PSICOTERAPIA, EXCLUIDOS_PSICOTERAPIA),
    "adicciones_integral": (GRUPOS_ADICCIONES_INTEGRAL, EXTRA_ADICCIONES_INTEGRAL, EXCLUIDOS_ADICCIONES_INTEGRAL),
    "marca_profesional": (GRUPOS_MARCA_PROFESIONAL, EXTRA_MARCA_PROFESIONAL, EXCLUIDOS_MARCA_PROFESIONAL),
}
DOMINIOS_ACADEMICOS = [d.split("/")[0] for g in GRUPOS_DOMINIOS for d in g] + DOMINIOS_EXTRA_ACEPTADOS


def usar_fuentes(nombre):
    global GRUPOS_DOMINIOS, DOMINIOS_ACADEMICOS, EXCLUIDOS_PERFIL
    grupos, extra, *resto = PERFILES_FUENTES[nombre]
    GRUPOS_DOMINIOS = grupos
    DOMINIOS_ACADEMICOS = [d.split("/")[0] for g in grupos for d in g] + extra
    EXCLUIDOS_PERFIL = re.compile(resto[0], re.I) if resto else None
MAX_PALABRAS_GOOGLE = 32

MODELOS = {
    "prueba": {"sintesis": "gpt-4o-mini", "preguntas": "gpt-4o-mini", "fuerte": "gpt-4o-mini"},
    "calidad": {"sintesis": "gpt-4o-mini", "preguntas": "gpt-4o-mini", "fuerte": "gpt-4o"},
    "avanzado": {"sintesis": "gpt-4o-mini", "preguntas": "gpt-6-sol", "fuerte": "gpt-6-sol"},
    "maximo": {"sintesis": "gpt-4o-mini", "preguntas": "gpt-6-sol", "fuerte": "gpt-6-astra"},
    "extenso": {"sintesis": "gpt-4o-mini", "preguntas": "gpt-6-sol", "fuerte": "gpt-6-sol"},
    "extenso-maximo": {"sintesis": "gpt-4o-mini", "preguntas": "gpt-6-sol", "fuerte": "gpt-6-astra"},
}
PERFILES = {
    "normal": {"max_conv_turn": 3, "max_perspective": 3, "search_top_k": 5, "retrieve_top_k": 3, "tokens_esquema": 400, "tokens_seccion": 700, "palabras_evidencia": 1500},
    "extenso": {"max_conv_turn": 5, "max_perspective": 6, "search_top_k": 6, "retrieve_top_k": 10, "tokens_esquema": 1000, "tokens_seccion": 1100, "palabras_evidencia": 4000},
}
PRECIOS = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00), "gpt-6-luna": (0.10, 0.50), "gpt-6-sol": (2.00, 10.00), "gpt-6-astra": (10.00, 50.00)}
FAMILIAS_RAZONAMIENTO = ("gpt-5", "gpt-6", "o1", "o3", "o4")
RESERVA_RAZONAMIENTO = {"none": 0, "minimal": 1000, "low": 3000, "medium": 8000, "high": 16000, "xhigh": 24000, "max": 32000}


def es_razonamiento(modelo):
    return modelo.startswith(FAMILIAS_RAZONAMIENTO)


class ModeloOpenAI(OpenAIModel):
    def __init__(self, model, max_tokens, esfuerzo="low", **kwargs):
        super().__init__(model=model, max_tokens=max_tokens, model_type="chat", **kwargs)
        self.razonamiento = es_razonamiento(model)
        self.esfuerzo = esfuerzo

    def _adaptar(self, kw):
        if not self.razonamiento:
            return kw
        kw = dict(kw)
        visibles = kw.pop("max_tokens", 500)
        for k in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
            kw.pop(k, None)
        kw["max_completion_tokens"] = visibles + RESERVA_RAZONAMIENTO.get(self.esfuerzo, 3000)
        kw["reasoning_effort"] = self.esfuerzo
        return kw

    def basic_request(self, prompt, **kwargs):
        raw_kwargs = kwargs
        kw = self._adaptar({**self.kwargs, **kwargs})
        messages = [{"role": "user", "content": prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        kw["messages"] = messages
        kw = {"stringify_request": json.dumps(kw)}
        response = chat_request(**kw)
        self.history.append({"prompt": prompt, "response": response, "kwargs": kw, "raw_kwargs": raw_kwargs})
        return response

INSTRUCCION_BASE = (
    "You are assisting in writing an academic, evidence-based text. "
    "Report the strength of the evidence accurately: do not overstate effectiveness, distinguish evidence-based from "
    "evidence-informed practices, and mention limitations of the research when the sources do. "
    "Always follow the exact output format requested in the user message."
)
INSTRUCCION_PROPOSITO = (
    "Purpose and intended audience, as stated by the editor: {proposito}\n"
    "Keep this purpose in mind when choosing perspectives, asking questions, organizing the outline and writing."
)
INSTRUCCION_EXTENSO = (
    "This text will become a long-form training module of about 3,500 words in total. "
    "When creating an outline, include 6 or 7 main sections, each with 2 or 3 subsections, without overlapping content between sections. "
    "When writing a section, keep it to about 450 to 550 words and focused on its own topic, without re-defining the general topic: develop every subsection of the outline "
    "in depth with definitions, mechanisms, findings from the collected information and implications for practice, "
    "citing the collected information for every factual claim. Do not pad with repetition."
)


def construir_sistema(proposito, extenso):
    partes = [INSTRUCCION_BASE]
    if proposito:
        partes.append(INSTRUCCION_PROPOSITO.format(proposito=proposito))
    if extenso:
        partes.append(INSTRUCCION_EXTENSO)
    return "\n\n".join(partes)


def ampliar_evidencia(palabras):
    original = ArticleTextProcessing.limit_word_count_preserve_newline

    def limite(texto, maximo):
        return original(texto, palabras if maximo == 1500 else maximo)

    ArticleTextProcessing.limit_word_count_preserve_newline = staticmethod(limite)


def costo_storm(lm_cost):
    total, detalle = 0.0, {}
    for modulo in lm_cost.values():
        for modelo, t in modulo.items():
            pe, ps = PRECIOS.get(modelo, (0, 0))
            c = t.get("prompt_tokens", 0) / 1e6 * pe + t.get("completion_tokens", 0) / 1e6 * ps
            detalle[modelo] = detalle.get(modelo, 0) + c
            total += c
    return total, detalle


def filtro_sitios(grupo):
    return "(" + " OR ".join(f"site:{d}" for d in grupo) + ")"


EXCLUIDOS_PERFIL = None
EXCLUIDOS = re.compile(r"/topics/|sk\.sagepub\.com/(ency|dict)|/careers/|/news/|/blog/|/press-?release|/newsroom/|/podcasts?/|/videos?/|/events?/|/stories/|/donate|/shop/|/store/|/membership/|/jobs/|wikipedia\.org|/media-centre/", re.I)


def es_academico(url):
    u = urlparse(url)
    host = (u.hostname or "").lower()
    if EXCLUIDOS.search(url) or (u.path.strip("/") == "" and "id=" not in u.query):
        return False
    if EXCLUIDOS_PERFIL is not None and EXCLUIDOS_PERFIL.search(url):
        return False
    return any(host == d or host.endswith("." + d) for d in DOMINIOS_ACADEMICOS)


class SerperAcademicoRM(SerperRM):
    def resolver_enlaces(self, resultado):
        if not isinstance(resultado, dict) or not any("google.com/goto" in (o.get("link") or "") for o in resultado.get("organic", [])):
            return resultado
        datos = resultado
        for _ in range(4):
            try:
                r = requests.post(
                    f"{self.base_url}/resolve",
                    headers={"X-API-KEY": self.serper_search_api_key, "Content-Type": "application/json"},
                    json=datos,
                    timeout=120,
                )
                datos = r.json()
            except Exception as e:
                print(f"Aviso: no se pudieron resolver los enlaces de Google ({e})", flush=True)
                return resultado
            if "retryAfter" not in datos:
                break
            time.sleep(min(float(datos.get("retryAfter") or 5), 60))
        return datos if isinstance(datos, dict) and "organic" in datos else resultado

    def serper_runner(self, query_params):
        resultado = super().serper_runner(query_params)
        if isinstance(resultado, dict) and "organic" not in resultado and ("message" in resultado or "error" in resultado):
            print(f"\nERROR DE SERPER: {resultado.get('message') or resultado.get('error')}. Revise la clave y los créditos en serper.dev. Se detiene la ejecución.", flush=True)
            os._exit(1)
        return self.resolver_enlaces(resultado)

    def forward(self, query_or_queries, exclude_urls=[]):
        consultas = [query_or_queries] if isinstance(query_or_queries, str) else list(query_or_queries)
        salida = []
        for q in consultas:
            palabras_q = q.split()
            por_grupo = []
            for grupo in GRUPOS_DOMINIOS:
                libres = MAX_PALABRAS_GOOGLE - (2 * len(grupo) - 1)
                texto = " ".join(palabras_q[:max(libres, 5)])
                try:
                    por_grupo.append(super().forward([f"{texto} {filtro_sitios(grupo)}"], exclude_urls))
                except Exception as e:
                    print(f"Aviso: búsqueda fallida ({e})")
                    por_grupo.append([])
            vistos, combinados = set(), []
            for i in range(max((len(x) for x in por_grupo), default=0)):
                for lista in por_grupo:
                    if i < len(lista):
                        r = lista[i]
                        url = r.get("url", "")
                        r["snippets"] = [t for t in (r.get("snippets") or []) if isinstance(t, str) and t.strip()]
                        if url and url not in vistos and r["snippets"] and es_academico(url):
                            vistos.add(url)
                            combinados.append(r)
            salida.extend(combinados[: self.k])
        return salida


def pedir_proposito(args):
    if args.proposito is not None:
        return args.proposito.strip()
    if not sys.stdin.isatty():
        return ""
    print("Propósito del artículo: para qué y para quién se escribe (Enter para omitir).")
    print("Ejemplo: Revisión para psicólogos clínicos de Latinoamérica sobre la evidencia y límites de los PAP.")
    return input("Propósito: ").strip()


def main(args):
    raiz = RAIZ
    load_api_key(toml_file_path=os.path.join(raiz, "secrets.toml"))
    faltan = [k for k in ("OPENAI_API_KEY", "SERPER_API_KEY") if not os.getenv(k)]
    if faltan:
        sys.exit(f"Faltan claves en secrets.toml: {', '.join(faltan)}")

    usar_fuentes(args.fuentes)
    proposito = pedir_proposito(args)
    extenso = args.modo.startswith("extenso")
    perfil = dict(PERFILES["extenso" if extenso else "normal"])
    for clave in ("max_conv_turn", "max_perspective", "search_top_k", "retrieve_top_k"):
        if getattr(args, clave) is not None:
            perfil[clave] = getattr(args, clave)
    if extenso:
        ampliar_evidencia(perfil["palabras_evidencia"])
    openai_kwargs = {"api_key": os.getenv("OPENAI_API_KEY"), "temperature": 1.0, "top_p": 0.9, "system_prompt": construir_sistema(proposito, extenso)}

    m = MODELOS[args.modo]
    e = {"esfuerzo": args.esfuerzo}
    lm_configs = STORMWikiLMConfigs()
    lm_configs.set_conv_simulator_lm(ModeloOpenAI(m["sintesis"], 500, **e, **openai_kwargs))
    lm_configs.set_question_asker_lm(ModeloOpenAI(m["preguntas"], 500, **e, **openai_kwargs))
    lm_configs.set_outline_gen_lm(ModeloOpenAI(m["fuerte"], perfil["tokens_esquema"], **e, **openai_kwargs))
    lm_configs.set_article_gen_lm(ModeloOpenAI(m["fuerte"], perfil["tokens_seccion"], **e, **openai_kwargs))
    lm_configs.set_article_polish_lm(ModeloOpenAI(m["fuerte"], 4000, **e, **openai_kwargs))

    engine_args = STORMWikiRunnerArguments(
        output_dir=os.path.join(raiz, "outputs", "articles"),
        max_conv_turn=perfil["max_conv_turn"],
        max_perspective=perfil["max_perspective"],
        search_top_k=perfil["search_top_k"],
        retrieve_top_k=perfil["retrieve_top_k"],
        max_thread_num=args.max_thread_num,
    )
    args.search_top_k = perfil["search_top_k"]
    rm = SerperAcademicoRM(
        serper_search_api_key=os.getenv("SERPER_API_KEY"),
        k=args.search_top_k,
        query_params={"autocorrect": True, "num": args.search_top_k, "page": 1},
    )

    print(f"Modo: {args.modo} | Modelos: {m} | Esfuerzo de razonamiento: {args.esfuerzo} | Tema: {args.topic}")
    print(f"Propósito: {proposito or '(sin propósito)'}")
    print(f"Perfil: {'extenso' if extenso else 'normal'} | fuentes: {args.fuentes} | {perfil}")
    runner = STORMWikiRunner(engine_args, lm_configs, rm)
    personas_extra = PERSONAS_EXTRA.get(args.fuentes, [])
    if personas_extra:
        generador = runner.storm_knowledge_curation_module.persona_generator
        generar_original = generador.generate_persona

        def generar_con_extras(topic, max_num_persona=3):
            return generar_original(topic=topic, max_num_persona=max_num_persona) + personas_extra

        generador.generate_persona = generar_con_extras
        print(f"Perspectivas adicionales fijas: {len(personas_extra)}")
    pulir_original = runner.run_article_polishing_module

    def pulir_y_guardar_referencias(*a, **k):
        articulo = pulir_original(*a, **k)
        articulo.dump_reference_to_file(os.path.join(runner.article_output_dir, "url_to_info_polished.json"))
        return articulo

    runner.run_article_polishing_module = pulir_y_guardar_referencias
    runner.run(
        topic=args.topic,
        do_research=True,
        do_generate_outline=True,
        do_generate_article=True,
        do_polish_article=True,
    )
    runner.post_run()
    runner.summary()
    total, detalle = costo_storm(runner.lm_cost)
    print(f"Coste estimado STORM: ≈ {total:.3f} USD " + " | ".join(f"{k}: {v:.3f}" for k, v in detalle.items()))

    carpeta = getattr(runner, "article_output_dir", None) or os.path.join(
        engine_args.output_dir, args.topic.replace(" ", "_").replace("/", "_")
    )
    with open(os.path.join(carpeta, "proposito.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"tema": args.topic, "proposito": proposito, "modo": args.modo, "modelos": m, "esfuerzo": args.esfuerzo, "perfil": perfil, "coste_storm_usd": round(total, 4), "estilo": args.estilo, "fuentes": args.fuentes, "fecha": datetime.now().isoformat(timespec="seconds")},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("\n***** Referencias APA 7 *****")
    try:
        procesar(carpeta, forzar=True)
    except Exception as e:
        print(f"No se pudieron generar las referencias: {e}")
        print(f"Puede reintentarlo con: python scripts/agregar_referencias.py \"{carpeta}\" --forzar")
        return

    espanol = args.espanol if args.espanol is not None else extenso
    if espanol:
        print("\n***** Adaptación didáctica al español *****")
        try:
            from adaptar_espanol import adaptar
            ruta_es = adaptar(carpeta, modelo=m["fuerte"] if m["fuerte"].startswith("gpt-6") else "gpt-6-sol", esfuerzo=args.esfuerzo, estilo=args.estilo)
        except Exception as e:
            print(f"No se pudo adaptar al español: {type(e).__name__}: {e}")
            print(f"Puede reintentarlo con: python scripts/adaptar_espanol.py \"{carpeta}\"")
            return
        if str(args.figuras) != "0":
            print("\n***** Figuras y tablas APA 7 *****")
            try:
                from generar_figuras import generar
                generar(carpeta, n=args.figuras, semilla=args.semilla)
            except Exception as e:
                print(f"No se pudieron generar las figuras: {type(e).__name__}: {e}")
                print(f"Puede reintentarlo con: python scripts/generar_figuras.py \"{carpeta}\"")
    if not args.sin_docx:
        print("\n***** Exportación a Word (APA 7) *****")
        try:
            from exportar_docx import exportar
            origen_md = ruta_es if espanol else os.path.join(carpeta, "articulo_apa.md")
            ruta_docx = exportar(origen_md, autor=args.autor, afiliacion=args.afiliacion, curso=args.curso, idioma="es-CO" if espanol else "en-US")
            print(f"Documento Word: {ruta_docx}")
            print("Al abrirlo, Word preguntará si desea actualizar los campos: responda Sí para generar el índice.")
        except Exception as e:
            print(f"No se pudo exportar a Word: {type(e).__name__}: {e}")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--topic", required=True)
    p.add_argument("--proposito", default=None)
    p.add_argument("--modo", choices=list(MODELOS), default="prueba")
    p.add_argument("--esfuerzo", choices=list(RESERVA_RAZONAMIENTO), default="low")
    p.add_argument("--max-conv-turn", type=int, default=None)
    p.add_argument("--max-perspective", type=int, default=None)
    p.add_argument("--search-top-k", type=int, default=None)
    p.add_argument("--retrieve-top-k", type=int, default=None)
    p.add_argument("--espanol", dest="espanol", action="store_true", default=None)
    p.add_argument("--sin-espanol", dest="espanol", action="store_false")
    p.add_argument("--sin-docx", action="store_true")
    p.add_argument("--autor", default="Instituto Salamanca")
    p.add_argument("--afiliacion", default="")
    p.add_argument("--curso", default="")
    p.add_argument("--fuentes", choices=list(PERFILES_FUENTES), default="salud", help="salud: psicología, medicina y salud pública; educacion: educación, e-learning, tecnología y normas de accesibilidad/usabilidad; vinculos: orientación y terapia de pareja y familia, ciclo vital, género, violencia y códigos éticos; organizacional: consejería laboral, psicología organizacional y de la salud ocupacional, riesgos psicosociales, desarrollo de carrera, normas OIT/ISO 45003 y códigos éticos; escolar: consejería y orientación educativa, orientación vocacional y proyecto de vida, bienestar y competencias socioemocionales, inclusión, marcos de UNESCO/UNICEF/OMS y estándares de la ASCA; adicciones: tratamiento psicoterapéutico y farmacológico de las adicciones, TCC, craving, intervención breve en atención primaria y guías clínicas de OMS/UNODC, SAMHSA, NIDA, NIAAA, NICE, EUDA y VA/DoD; psicoterapia: modelos psicoterapéuticos (TCC de primera, segunda y tercera generación, sistémicos, integradores, centrados en la persona), factores comunes y alianza terapéutica, terapia en línea y competencias digitales, con guías de OMS/NICE/APA y marcos éticos y normativos de EFPA, IUPsyS, COP, Colpsic, ISO 13131, RGPD y ASPPB; adicciones_integral: abordaje biopsicosocial integral de las adicciones a sustancias y comportamentales (neurobiología, alcohol, depresores, estimulantes, opioides, cannabis, alucinógenos y disociativos, adicciones digitales y juego patológico), guías clínicas de OMS/UNODC, SAMHSA, NIDA, NIAAA, NICE, EUDA, ASAM y VA/DoD, y marcos regulatorios internacionales de JIFE/INCB, ONU, CICAD, FDA, EMA, CDC y RGPD/UE; marca_profesional: identidad profesional, marca personal, posicionamiento, reputación y presencia digital del consejero, y ética de la promoción de servicios (confidencialidad, testimonios, casos, afirmaciones sobre resultados, alcance de la consejería), con revistas de consejería y psicología profesional, literatura académica de marca personal y profesionalismo digital, códigos de ACA, NBCC, BACP, CCPA, COP y Colpsic, y normas de publicidad de FTC, ASA/CAP y RGPD")
    p.add_argument("--estilo", choices=["modulo", "articulo"], default="modulo", help="modulo: con objetivos, puntos clave, caso y preguntas; articulo: prosa continua con conclusiones")
    p.add_argument("--max-thread-num", type=int, default=3)
    p.add_argument("--figuras", default="auto", help="figuras y tablas APA en la versión en español: «auto» (predeterminado, entre 2 y 6 según el contenido), un número fijo, o 0 para desactivarlas")
    p.add_argument("--semilla", type=int, default=None, help="semilla del orden aleatorio del catálogo de figuras, para reproducir un resultado")
    main(p.parse_args())
