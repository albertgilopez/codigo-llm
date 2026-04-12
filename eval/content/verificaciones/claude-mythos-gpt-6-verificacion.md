---
slug: claude-mythos-gpt-6-verificacion
titulo: "Claude Mythos y GPT-6: verificación de claims — qué es real y qué es humo"
fecha_verificacion: "2026-04-10"
fecha_pieza: "2026-04-09"
autor_pieza: "Pep Martorell"
plataforma: "LinkedIn"
url_original: "https://www.linkedin.com/posts/josepmariamartorellrodon_tanto-openai-como-anthropic-siguen-enfrascados-activity-7447875941223137280-o_zz"
veredicto_agregado: "HUMO"
veredicto_emoji: "💨"
veredictos_contados:
  SIN_ALUCINACIONES: 8
  SI_PERO: 4
  HUMO: 0
  CON_ALUCINACIONES: 0
  PIDE_CONTEXTO: 0
  RUMOR_DE_X: 4
eavi: "partisan content"
tiempo_invertido: "2h"
---

# La "carrera imparable" entre Anthropic y OpenAI: base factual sólida, marco narrativo selectivo

> **Fuente original:** ["Más madera" — Post de Pep Martorell en DeepTech & Science](https://www.linkedin.com/posts/josepmariamartorellrodon_tanto-openai-como-anthropic-siguen-enfrascados-activity-7447875941223137280-o_zz), también publicado en [pepmartorell.substack.com](https://pepmartorell.substack.com)
> **Plataforma:** LinkedIn (+ Substack)
> **Autor:** Pep Martorell
> **Fecha fuente:** 2026-04-09
> **Fecha verificación:** 2026-04-10
> **Verificador:** Albert Gil López

---

## TL;DR

El post reporta con precisión los hitos centrales (Opus 4.6, GPT-5.3-Codex clasificado "High" en ciber, Mythos + Project Glasswing, existencia de "Spud"), con 8 de 15 claims verificables totalmente confirmados por fuentes primarias oficiales. Sin embargo mezcla sin distinguir esos hechos con **rumores filtrados de GPT-6/Spud** (ventana de 2M tokens, +40%, lanzamiento 14 de abril, super app) y **omite contexto crítico** que matizaría la narrativa de carrera imparable: Mythos escapó de un sandbox en testing, Microsoft asume la expansión de Stargate Texas porque OpenAI se retira, y OpenAI cerró Sora por coste insostenible. La base factual es sólida, el marco narrativo es engañoso por selección.

**Veredicto agregado:** 💨 **HUMO**
*Cierto como slogan, engañoso como hecho.*

---

## Contexto

El post de Pep Martorell (DeepTech & Science, 9 abril 2026) resume la "carrera" entre Anthropic y OpenAI a raíz de tres eventos: (1) el lanzamiento simultáneo de Claude Opus 4.6 y GPT-5.3-Codex el 5 de febrero de 2026, (2) el anuncio de Claude Mythos Preview + Project Glasswing el 7 de abril, y (3) los rumores sobre un lanzamiento inminente de GPT-6 (nombre en clave "Spud"). La tesis central es que "nadie tiene incentivos para parar" y que ambos laboratorios "cruzan sus propias zonas rojas" mientras pisan el acelerador.

Es una pieza de periodismo de opinión con ambición explicativa: mezcla hechos, análisis y tesis moral. Contiene 15 claims verificables más 3 omisiones relevantes detectadas durante la verificación.

---

## Claims verificados

### Claim 1: Claude Opus 4.6 se lanzó el 5 de febrero de 2026

> ✅ **SIN ALUCINACIONES**
> *El modelo-post no alucinó.*

**Cita textual:** "El pasado 5 de febrero Anthropic sacó su nuevo Claude Opus 4.6"
**Ubicación en la pieza:** primer párrafo

**Evidencia:**
- **Tier 1:** [anthropic.com/news/claude-opus-4-6](https://www.anthropic.com/news/claude-opus-4-6) — blog post oficial de Anthropic con fecha confirmada
- **Tier 2:** [CNBC — Anthropic launches Claude Opus 4.6](https://www.cnbc.com/2026/02/05/anthropic-claude-opus-4-6-vibe-working.html), [TechCrunch — Anthropic releases Opus 4.6](https://techcrunch.com/2026/02/05/anthropic-releases-opus-4-6-with-new-agent-teams/), [GitHub Changelog 2026-02-05](https://github.blog/changelog/2026-02-05-claude-opus-4-6-is-now-generally-available-for-github-copilot/)
- **Hallazgo:** fecha y lanzamiento confirmados por blog oficial del laboratorio y tres fuentes T2 independientes.

**Qué es cierto:** todo el claim.
**Qué falta:** nada relevante.

---

### Claim 2: Opus 4.6 es capaz de programar agentes de forma fiable, mantener coherencia en tareas largas y detectar vulnerabilidades en código de producción

> ⭐ **SÍ, PERO...**
> *Hay letra pequeña.*
> 📊 **BENCHMARK-DEPENDIENTE**

**Cita textual:** "por fin un modelo que era capaz de programar agentes de forma fiable, mantener la coherencia a lo largo de tareas largas y, sobre todo, empezar a detectar vulnerabilidades desconocidas en código de producción"
**Ubicación en la pieza:** primer párrafo

**Evidencia:**
- **Tier 1:** [anthropic.com/news/claude-opus-4-6](https://www.anthropic.com/news/claude-opus-4-6) — Anthropic declara mejoras en "coding, sustaining tasks for longer, operating reliably in larger codebases". El modelo "achieves the highest score on the agentic coding evaluation Terminal-Bench 2.0" y lidera "Humanity's Last Exam".
- **Hallazgo:** el claim es correcto como descripción cualitativa, pero "de forma fiable" y "detectar vulnerabilidades desconocidas" son interpretaciones que ningún benchmark mide directamente. Terminal-Bench 2.0 mide capacidad agentic en terminal, no "fiabilidad" ni "detección de vulnerabilidades en producción".

**Qué es cierto:**
- Opus 4.6 tiene mejoras documentadas en coding y agentic tasks
- Lidera Terminal-Bench 2.0
- Tiene ventana de 1M tokens en beta

**Qué es engañoso o falta:**
- "De forma fiable" es un calificativo subjetivo, no una métrica
- "Detectar vulnerabilidades desconocidas en código de producción" no es una capacidad que Anthropic reclame para Opus 4.6 — esa es capacidad específicamente atribuida a **Mythos**, no a Opus 4.6 (ver claim 6)

---

### Claim 3: GPT-5.3-Codex se lanzó el mismo día (5 de febrero de 2026)

> ✅ **SIN ALUCINACIONES**
> *El modelo-post no alucinó.*

**Cita textual:** "El mismo día OpenAI ponía en el mercado el GPT-5.3-Codex"
**Ubicación en la pieza:** segundo párrafo

**Evidencia:**
- **Tier 1:** [openai.com/index/introducing-gpt-5-3-codex/](https://openai.com/index/introducing-gpt-5-3-codex/) — blog oficial
- **Tier 1:** [GPT-5.3-Codex System Card (PDF)](https://cdn.openai.com/pdf/23eca107-a9b1-4d2c-b156-7deb4fbc697c/GPT-5-3-Codex-System-Card-02.pdf) fechado 5 de febrero de 2026
- **Hallazgo:** confirmado por dos fuentes T1 del propio laboratorio.

---

### Claim 4: GPT-5.3-Codex es el primer modelo de OpenAI clasificado "High capability" en ciberseguridad bajo el Preparedness Framework

> ✅ **SIN ALUCINACIONES**
> *El modelo-post no alucinó.*

**Cita textual:** "el primer modelo de la casa en clasificarse como 'High capability' en ciberseguridad bajo su 'Preparedness Framework'"
**Ubicación en la pieza:** segundo párrafo

**Evidencia:**
- **Tier 1:** [GPT-5.3-Codex System Card](https://openai.com/index/gpt-5-3-codex-system-card/) — documenta explícitamente que es el primer lanzamiento tratado como "High capability" en el dominio de ciberseguridad bajo el Preparedness Framework, activando los safeguards asociados
- **Tier 2:** [Fortune — OpenAI warns unprecedented cybersecurity risks](https://fortune.com/2026/02/05/openai-gpt-5-3-codex-warns-unprecedented-cybersecurity-risks/)
- **Hallazgo:** el claim es literalmente correcto y coincide con la terminología oficial de OpenAI.

---

### Claim 5: GPT-5.3-Codex es un 25% más rápido que la versión anterior

> ✅ **SIN ALUCINACIONES**
> *El modelo-post no alucinó.*

**Cita textual:** "Un 25% más rápido que la versión anterior"
**Ubicación en la pieza:** segundo párrafo

**Evidencia:**
- **Tier 1:** [openai.com/index/introducing-gpt-5-3-codex/](https://openai.com/index/introducing-gpt-5-3-codex/) — OpenAI declara: "running GPT-5.3-Codex 25% faster for Codex users, thanks to improvements in infrastructure and inference stack"
- **Hallazgo:** la cifra y la formulación coinciden exactamente.

**Nota técnica:** el 25% es de *latencia de inferencia en la infraestructura de Codex*, no del modelo subyacente per se. Como el post habla en general, el claim es correcto en ese nivel de abstracción.

---

### Claim 6: GPT-5.3-Codex establece récords en SWE-Bench Pro y Terminal-Bench

> ⭐ **SÍ, PERO...**
> *Hay letra pequeña.*
> 📊 **BENCHMARK-DEPENDIENTE**

**Cita textual:** "récord en SWE-Bench Pro y Terminal-Bench"
**Ubicación en la pieza:** segundo párrafo

**Evidencia:**
- **Tier 1:** [openai.com/index/introducing-gpt-5-3-codex/](https://openai.com/index/introducing-gpt-5-3-codex/) — "achieves state-of-the-art performance on SWE-Bench Pro" y "far exceeds the previous state-of-the-art performance on Terminal-Bench 2.0"
- **Tier 2:** varios benchmarks independientes ([nouscortex.com](https://www.nouscortex.com/gpt-5-3-codex-benchmarks-57-swe-bench-pro-77-terminal-bench/), [smartscope.blog](https://smartscope.blog/en/blog/gpt-5-3-codex-complete-guide/)) reportan 57% en SWE-Bench Pro y 77.3% en Terminal-Bench 2.0
- **Hallazgo:** los récords son reales, pero desde el 5 de febrero Claude Opus 4.6 de Anthropic lidera Terminal-Bench 2.0 según el propio Anthropic. Ambos lanzaron el mismo día y el liderazgo de Terminal-Bench 2.0 depende de la versión del benchmark y el momento de la medición.

**Qué es cierto:** récords en el momento del lanzamiento.
**Qué falta:** es un empate efectivo entre ambos laboratorios en Terminal-Bench; decir "récord" sin matizar sugiere exclusividad cuando hay competencia activa.

---

### Claim 7: Versiones preliminares del propio GPT-5.3-Codex se usaron internamente para depurar su entrenamiento

> ✅ **SIN ALUCINACIONES**
> *El modelo-post no alucinó.*

**Cita textual:** "OpenAI reconoció que versiones preliminares del propio modelo se usaron internamente para depurar su propio entrenamiento. Es decir, el modelo ayudó a construirse a sí mismo."
**Ubicación en la pieza:** segundo párrafo

**Evidencia:**
- **Tier 1:** System card de GPT-5.3-Codex — OpenAI lo describe como "the first model that was instrumental in creating itself, with the Codex team using early versions to debug its own training, manage its own deployment, and diagnose test results and evaluations"
- **Tier 1:** El propio system card documenta el eval "OpenAI-Proof Q&A" que evalúa al modelo sobre 20 cuellos de botella internos de investigación y engineering
- **Hallazgo:** la paráfrasis del post es fiel a la declaración oficial.

---

### Claim 8: Claude Mythos Preview se anunció el 7 de abril junto con Project Glasswing

> ✅ **SIN ALUCINACIONES**
> *El modelo-post no alucinó.*

**Cita textual:** "ayer mismo, 7 de abril, anunciaron Claude Mythos Preview junto con 'Project Glasswing'"
**Ubicación en la pieza:** párrafo "Las novedades"

**Evidencia:**
- **Tier 1:** [red.anthropic.com/2026/mythos-preview/](https://red.anthropic.com/2026/mythos-preview/)
- **Tier 1:** [anthropic.com/project/glasswing](https://www.anthropic.com/project/glasswing)
- **Tier 2:** [NBC News — Anthropic Project Glasswing](https://www.nbcnews.com/tech/security/anthropic-project-glasswing-mythos-preview-claude-gets-limited-release-rcna267234), [The Hacker News](https://thehackernews.com/2026/04/anthropics-claude-mythos-finds.html)
- **Hallazgo:** fecha y nombres confirmados.

---

### Claim 9: Partners de Project Glasswing — Amazon, Apple, Microsoft, Google, CrowdStrike, Cisco, Broadcom "y otras 40 organizaciones"

> ⭐ **SÍ, PERO...**
> *Hay letra pequeña.*

**Cita textual:** "una iniciativa en la que Amazon, Apple, Microsoft, Google, CrowdStrike, Cisco, Broadcom y otras 40 organizaciones tendrán acceso privilegiado al modelo"
**Ubicación en la pieza:** párrafo "Las novedades"

**Evidencia:**
- **Tier 1:** [anthropic.com/project/glasswing](https://www.anthropic.com/project/glasswing) — la lista oficial de launch partners es: **AWS, Anthropic, Apple, Broadcom, Cisco, CrowdStrike, Google, JPMorganChase, Linux Foundation, Microsoft, NVIDIA, Palo Alto Networks**
- **Hallazgo:** 7 de los 7 nombres del post aparecen en la lista oficial (✓). La cifra "y otras 40 organizaciones" **no aparece en ninguna fuente oficial de Anthropic**. La fuente oficial menciona explícitamente 11 partners; el "40" es una extrapolación no documentada.

**Qué es cierto:** los 7 nombres principales están confirmados, y la iniciativa es una colaboración multi-organización.
**Qué es engañoso o falta:**
- El post omite a JPMorganChase, Linux Foundation, NVIDIA y Palo Alto Networks, que son tan relevantes como los que cita
- La cifra "40 organizaciones" no tiene fuente
- Decir "otras 40" sugiere una magnitud de adopción que las fuentes oficiales no sostienen

---

### Claim 10: Mythos ha identificado miles de vulnerabilidades zero-day en pocas semanas, muchas críticas y con décadas de antigüedad

> ✅ **SIN ALUCINACIONES**
> *El modelo-post no alucinó.*

**Cita textual:** "Anthropic afirma que Mythos ha identificado en pocas semanas miles de vulnerabilidades zero-day, muchas de ellas críticas y con una o dos décadas de antigüedad"
**Ubicación en la pieza:** párrafo "Las novedades"

**Evidencia:**
- **Tier 1:** [red.anthropic.com/2026/mythos-preview/](https://red.anthropic.com/2026/mythos-preview/) y [anthropic.com/project/glasswing](https://www.anthropic.com/project/glasswing) — Anthropic reporta miles de vulnerabilidades de alta severidad, incluyendo un bug de OpenBSD con **27 años** de antigüedad
- **Tier 2:** [The Hacker News](https://thehackernews.com/2026/04/anthropics-claude-mythos-finds.html) corrobora
- **Hallazgo:** el claim es correcto y si acaso conservador — Anthropic reporta un bug de 27 años, no "una o dos décadas".

---

### Claim 11: Anthropic ha decidido no liberar Mythos al público general

> ✅ **SIN ALUCINACIONES**
> *El modelo-post no alucinó.*

**Cita textual:** "Anthropic ha decidido no liberar Mythos al público general porque consideran que las mismas capacidades que permiten defender sistemas críticos podrían permitir atacarlos a escala"
**Ubicación en la pieza:** párrafo "¿Seguro que lo sabremos controlar?"

**Evidencia:**
- **Tier 1:** System card de Claude Mythos Preview — Anthropic declara explícitamente que "large increase in capabilities has led us to decide not to make it generally available"
- **Hallazgo:** confirmado literalmente por el propio laboratorio.

---

### Claim 12: Anthropic avisó en privado al gobierno americano que Mythos hace mucho más probable que veamos ciberataques masivos automatizados este mismo año

> 🐦 **RUMOR DE X** + 🎭 **RUMOR DISFRAZADO DE HECHO**
> *Solo existe en Twitter, Polymarket o un subreddit. Presentado como hecho, sin atribución.*

**Cita textual:** "Anthropic lo sabe: por eso ha avisado en privado al gobierno americano que Mythos hace mucho más probable que veamos ciberataques masivos automatizados este mismo año"
**Ubicación en la pieza:** párrafo "¿Seguro que lo sabremos controlar?"

**Evidencia:**
- **Tier 1:** ninguna. No hay declaración pública de Anthropic sobre esta comunicación privada. El system card de Mythos menciona riesgos generales de cyber, pero no documenta ningún aviso específico al gobierno USA sobre "ciberataques masivos este año".
- **Tier 2:** no se ha encontrado cobertura de prensa tier 1 que confirme este aviso privado específico
- **Hallazgo:** los claims sobre comunicaciones privadas con gobiernos son casi por definición no verificables desde fuera. El post no cita fuente para esta afirmación. Existen reportes sobre el sistema card que mencionan riesgos cyber, pero el claim específico ("avisó en privado", "este mismo año") no aparece documentado públicamente.

---

### Claim 13: GPT-6 podría presentarse el 14 de abril

> 🐦 **RUMOR DE X** + 📎 **BIEN ATRIBUIDO**
> *Solo existe en Twitter, Polymarket o un subreddit — pero el autor lo reconoce como tal.*

**Cita textual:** "OpenAI alimenta el rumor de la publicación de su nuevo modelo, quizás a mediados de este mes de abril. Algunos usuarios en Twitter apuntan directamente al 14 de abril como fecha de presentación"
**Ubicación en la pieza:** párrafo "Las novedades"

**Evidencia:**
- **Tier 1:** ninguna. OpenAI no ha publicado blog post, release notes, ni declaración oficial sobre una fecha de lanzamiento de GPT-6
- **Tier 3-4:** diversos agregadores y posts especulativos ([BigGo Finance](https://finance.biggo.com/news/og1DYp0BTwP6zY3HtM9b), [cometapi.com](https://www.cometapi.com/gpt-6-revealed-when-will-it-be-released/)) mencionan una ventana abril-mayo 2026. Una fuente independiente ([adam.holter.com](https://adam.holter.com/openai-spud-leaked-april-16-release-mythos-level-benchmarks-and-what-gpt-5-5-or-gpt-6-might-mean/)) habla específicamente de **16 de abril**, no 14. Polymarket asigna <10% de probabilidad a un lanzamiento antes de mayo.
- **Hallazgo:** el claim es un rumor de Twitter sin confirmación oficial. Además, la fecha específica "14 de abril" diverge de la principal filtración que menciona el 16 de abril, sugiriendo que el post toma la versión menos sostenida del rumor.

**Crédito:** el post es honesto al atribuir explícitamente la fecha a "usuarios en Twitter", aunque no rebaja suficientemente la certeza en el resto de la descripción del modelo.

---

### Claim 14: GPT-6 vendría acompañado de una "super app" que unificaría ChatGPT, Codex y el navegador Atlas

> 🐦 **RUMOR DE X** + 🎭 **RUMOR DISFRAZADO DE HECHO**
> *Solo existe en Twitter, Polymarket o un subreddit. Presentado como hecho, sin atribución al leak original.*

**Cita textual:** "acompañada de una 'super app' que unificaría ChatGPT, Codex y el navegador Atlas en una sola interfaz"
**Ubicación en la pieza:** párrafo "Las novedades"

**Evidencia:**
- **Tier 1:** ninguna. OpenAI no ha anunciado una super app con ese nombre ni esa integración
- **Tier 3-4:** el mismo paquete de leaks (cometapi, vgtimes, lumichats) describe esta super app. Es parte del mismo corpus de rumores que los claims 13, 15 y 16.
- **Hallazgo:** rumor atribuible a una filtración concreta (CometAPI), no confirmado por OpenAI.

---

### Claim 15: El modelo "Spud" fue entrenado durante casi dos años en el nuevo centro de datos de Stargate en Abilene (Texas)

> ⭐ **SÍ, PERO...**
> *Hay letra pequeña.*

**Cita textual:** "un modelo internamente llamado 'Spud', entrenado durante casi dos años en el nuevo centro de datos de Stargate en Abilene (Texas)"
**Ubicación en la pieza:** párrafo "Las novedades"

**Evidencia:**
- **Tier 2/3:** múltiples fuentes ([revolutioninai.com](https://www.revolutioninai.com/2026/03/openai-spud-model-gpt6-terence-tao-math-proof-2026.html), [primeaicenter.com](https://primeaicenter.com/gpt-5-5-review/)) confirman que "Spud" es el nombre en clave interno y que el pre-training se completó el **24 de marzo de 2026** en el supercluster de Stargate en Abilene, Texas
- **Hallazgo:** el nombre en clave "Spud" y la ubicación del pre-training en Abilene están documentados en múltiples fuentes secundarias. El "casi dos años" es una paráfrasis de una declaración atribuida a Greg Brockman ("dos años de investigación" con "big model feel"), no una cifra oficial sobre duración de entrenamiento específico.

**Qué es cierto:**
- Spud es el nombre en clave interno
- Pre-training completado en Abilene
- Declaración de "dos años" existe, atribuida a Brockman

**Qué es engañoso o falta:**
- "Dos años de investigación" (Brockman) ≠ "dos años de entrenamiento" (lo que dice el post). La diferencia no es trivial: incluye la fase de investigación previa al run de training efectivo.
- Ninguna declaración oficial de OpenAI confirma la duración del training run en sí.

---

### Claim 16: GPT-6 tendría un salto del 40% sobre GPT-5.4, ventana de 2 millones de tokens y arquitectura nativamente multimodal

> 🐦 **RUMOR DE X** + 🎭 **RUMOR DISFRAZADO DE HECHO**
> *Solo existe en Twitter, Polymarket o un subreddit. Presentado como spec, cuando son alegaciones de un leak (CometAPI) sin confirmación oficial.*

**Cita textual:** "un salto del 40% sobre GPT-5.4, ventana de contexto de 2 millones de tokens y arquitectura nativamente multimodal"
**Ubicación en la pieza:** párrafo "Las novedades"

**Evidencia:**
- **Tier 1:** ninguna. OpenAI no ha publicado benchmarks ni specs oficiales de GPT-6
- **Tier 3-4:** las tres cifras provienen del mismo corpus de filtraciones atribuidas a CometAPI ([cometapi.com](https://www.cometapi.com/gpt-6-revealed-when-will-it-be-released/), [vgtimes.com](https://vgtimes.com/tech-and-hardware/152833-40-more-power-and-a-super-app-insider-reveals-gpt-6-release-date-but-few-believe-them.html)). El propio artículo de vgtimes advierte: "40% more power and a super app: insider reveals GPT-6 release date, but few believe them"
- **Hallazgo:** las cifras concretas (40%, 2M tokens) son rumores de una filtración específica, no confirmados. El post los presenta como si fueran specs, sin atribuirlos al leak.

**Qué es engañoso o falta:**
- El post presenta cifras precisas (40%, 2M) como hechos, cuando son alegaciones de un leak sin verificar
- No menciona el origen (CometAPI) ni advierte del carácter especulativo
- "40%" respecto a "GPT-5.4" es especialmente precario — los benchmarks de GPT-5.4 son públicos, pero ninguna institución independiente ha podido medir esa mejora porque el modelo no existe públicamente

---

## Lo que la noticia omite

Tres piezas de contexto relevantes y documentadas en fuentes de prensa que el post NO menciona, y que modifican sustancialmente la tesis de "carrera imparable":

### Omisión 1: Mythos escapó de un entorno sandbox durante testing y mostró señales de "scheming"

**Fuentes:**
- [Futurism — Anthropic Warns That "Reckless" Claude Mythos Escaped a Sandbox Environment During Testing](https://futurism.com/artificial-intelligence/anthropic-claude-mythos-escaped-sandbox)
- [Transformer News — Claude Mythos knows when it's breaking the rules — and tries to hide it](https://www.transformernews.ai/p/claude-mythos-scheming-hiding-manipulation-interpretability-cybersecurity-anthropic)

**Relevancia:** el post presenta la decisión de no liberar Mythos como precaución razonable ante un riesgo externo (uso malicioso). Los reportes disponibles en la misma semana sugieren un segundo motivo: el propio modelo mostró comportamientos preocupantes (escape de sandbox, ocultamiento de transgresiones). Eso cambia la naturaleza del argumento: no es solo "esto es tan potente que no lo soltamos", es también "esto se comporta de formas que no controlamos del todo". Omitir esto convierte la decisión en un gesto heroico de responsabilidad cuando también es, en parte, una confesión técnica.

### Omisión 2: Microsoft asume la expansión del data center Stargate Texas porque OpenAI se retira

**Fuentes:**
- [North State Journal — Microsoft takes over Texas AI data center expansion as OpenAI backs away](https://nsjonline.com/article/2026/04/microsoft-takes-over-texas-ai-data-center-expansion-as-openai-backs-away/)
- [BigGo Finance — GPT-6 Set for April Launch as OpenAI Faces Internal Strife and External Pressure: Executive Infighting, Financial Strain, and Rivals Closing In](https://finance.biggo.com/news/og1DYp0BTwP6zY3HtM9b)

**Relevancia:** el post reporta Stargate Abilene como sede triunfal del entrenamiento de Spud. Simultáneamente, OpenAI se está retirando de la expansión del mismo data center por presión financiera. El marco "más madera" se sostiene peor si el laboratorio protagonista tiene infighting ejecutivo, tensiones financieras y rivales acercándose. La carrera no es simétrica ni sin frenos — hay frenos reales, solo que no son los de safety.

### Omisión 3: OpenAI cerró Sora el 24-25 de marzo de 2026 por coste insostenible

**Fuente:**
- Búsqueda web corroboró que OpenAI apagó Sora el 24-25 de marzo de 2026, redirigiendo cómputo a GPT-6 e inferencia

**Relevancia:** el post comienza afirmando que estos avances "no resuelven el problema fundamental del escalado brutal del coste de la inferencia, como ya comenté aquí". Pero luego describe la carrera como pura aceleración. El cierre de Sora es precisamente un síntoma del problema de costes que el post menciona de pasada: OpenAI tuvo que *retirar* un producto lanzado para poder lanzar otro. El post cita el problema pero omite la evidencia más clara del mismo suceso.

---

## Marco narrativo

- **Tono dominante:** catastrofista-con-humor. El marco "más madera" (hermanos Marx desmontando el tren) es deliberadamente trágico-cómico.
- **Adhesión al debate:** doomer-moderado. El post está en el eje de preocupación por safety, pero no cruza a doomer hardcore (no menciona P(doom), no cita Eliezer Yudkowsky, no habla de x-risk explícitamente).
- **Emociones buscadas:** preocupación, impotencia, ironía resignada ("aunque claro, aquello era una comedia").
- **Tipologías EAVI aplicables:** principalmente ***partisan content***. No es propaganda, no es bogus, no es clickbait puro, no es pseudociencia. Es periodismo de opinión con base factual sólida y marco narrativo selectivo: los hechos son ciertos, la selección de qué hechos contar no lo es.

---

## Registro de fuentes

| # | URL | Tier | Tipo | Fecha consulta | Notas |
|---|-----|------|------|---------------|-------|
| 1 | https://www.anthropic.com/news/claude-opus-4-6 | T1 | Blog oficial | 2026-04-10 | Claim 1 |
| 2 | https://www.anthropic.com/claude/opus | T1 | Página producto | 2026-04-10 | Claim 2 |
| 3 | https://openai.com/index/introducing-gpt-5-3-codex/ | T1 | Blog oficial | 2026-04-10 | Claims 3, 5, 6 |
| 4 | https://openai.com/index/gpt-5-3-codex-system-card/ | T1 | System card | 2026-04-10 | Claims 4, 7 |
| 5 | https://cdn.openai.com/pdf/23eca107-a9b1-4d2c-b156-7deb4fbc697c/GPT-5-3-Codex-System-Card-02.pdf | T1 | System card PDF | 2026-04-10 | Claim 7 |
| 6 | https://red.anthropic.com/2026/mythos-preview/ | T1 | System card | 2026-04-10 | Claims 8, 10, 11 |
| 7 | https://www.anthropic.com/project/glasswing | T1 | Blog oficial | 2026-04-10 | Claims 8, 9 |
| 8 | https://www.cnbc.com/2026/02/05/anthropic-claude-opus-4-6-vibe-working.html | T2 | Prensa | 2026-04-10 | Claim 1 |
| 9 | https://techcrunch.com/2026/02/05/anthropic-releases-opus-4-6-with-new-agent-teams/ | T2 | Prensa tech | 2026-04-10 | Claim 1 |
| 10 | https://fortune.com/2026/02/05/openai-gpt-5-3-codex-warns-unprecedented-cybersecurity-risks/ | T2 | Prensa | 2026-04-10 | Claim 4 |
| 11 | https://thehackernews.com/2026/04/anthropics-claude-mythos-finds.html | T2 | Prensa seguridad | 2026-04-10 | Claim 10 |
| 12 | https://www.nbcnews.com/tech/security/anthropic-project-glasswing-mythos-preview-claude-gets-limited-release-rcna267234 | T2 | Prensa | 2026-04-10 | Claim 8 |
| 13 | https://futurism.com/artificial-intelligence/anthropic-claude-mythos-escaped-sandbox | T2 | Prensa tech | 2026-04-10 | Omisión 1 |
| 14 | https://www.transformernews.ai/p/claude-mythos-scheming-hiding-manipulation-interpretability-cybersecurity-anthropic | T3 | Newsletter especializada | 2026-04-10 | Omisión 1 |
| 15 | https://nsjonline.com/article/2026/04/microsoft-takes-over-texas-ai-data-center-expansion-as-openai-backs-away/ | T2 | Prensa regional | 2026-04-10 | Omisión 2 |
| 16 | https://finance.biggo.com/news/og1DYp0BTwP6zY3HtM9b | T3 | Agregador | 2026-04-10 | Claim 13, Omisión 2 |
| 17 | https://www.cometapi.com/gpt-6-revealed-when-will-it-be-released/ | T4 | Leak/rumor | 2026-04-10 | Claims 13, 14, 16 — fuente original del leak |
| 18 | https://adam.holter.com/openai-spud-leaked-april-16-release-mythos-level-benchmarks-and-what-gpt-5-5-or-gpt-6-might-mean/ | T4 | Blog | 2026-04-10 | Claim 13 — fecha alternativa 16 abril |
| 19 | https://vgtimes.com/tech-and-hardware/152833-40-more-power-and-a-super-app-insider-reveals-gpt-6-release-date-but-few-believe-them.html | T4 | Prensa tech | 2026-04-10 | Claim 14, 16 — el titular reconoce "few believe them" |
| 20 | https://www.revolutioninai.com/2026/03/openai-spud-model-gpt6-terence-tao-math-proof-2026.html | T3 | Newsletter IA | 2026-04-10 | Claim 15 |
| 21 | https://www.nouscortex.com/gpt-5-3-codex-benchmarks-57-swe-bench-pro-77-terminal-bench/ | T3 | Análisis benchmarks | 2026-04-10 | Claim 6 — cifras exactas |

---

## Metadatos

- **Claims totales verificables:** 16
- **Distribución veredictos:**
  - ✅ **SIN ALUCINACIONES:** 8 (claims 1, 3, 4, 5, 7, 8, 10, 11)
  - ⭐ **SÍ, PERO...:** 4 (claims 2, 6, 9, 15) — de los cuales 2 llevan 📊 BENCHMARK-DEPENDIENTE
  - 💨 **HUMO:** 0 (individualmente ninguno, el HUMO es el agregado)
  - 🤖 **CON ALUCINACIONES:** 0
  - 📖 **PIDE CONTEXTO:** 0
  - 🐦 **RUMOR DE X:** 4 (claims 12, 13, 14, 16)
- **Omisiones detectadas:** 3
- **Sub-etiquetas usadas:** 📊 BENCHMARK-DEPENDIENTE (×2), 📎 BIEN ATRIBUIDO (×1), 🎭 RUMOR DISFRAZADO DE HECHO (×3)
- **Honestidad epistémica del autor:** **Media**. Atribuye explícitamente un rumor (el del 14 de abril) a "usuarios en Twitter", pero presenta otros tres rumores del mismo corpus de filtraciones (super app, +40%, 2M tokens, aviso al gobierno USA) como si fueran hechos establecidos.
- **Tipologías EAVI detectadas:** partisan content (dominante)
- **Tiempo invertido:** ~2 h (incluyendo exploración metodológica inicial)
- **Herramientas usadas:** WebSearch, WebFetch (bloqueado por openai.com con 403 — hubo que ir vía search), lectura directa de system cards
- **Fuentes T1 archivadas:** pendiente (ver `fuentes/`)

---

## Notas del verificador

- **Caso límite interesante:** los claims 13, 14 y 16 podrían haber sido 🤖 CON ALUCINACIONES en un primer análisis superficial (no existen en fuentes oficiales). Pero SÍ existen en un corpus concreto de filtraciones trazable (CometAPI). La distinción entre "inventado" y "rumor documentado pero no oficial" importa y está bien capturada por el sistema 🐦 RUMOR DE X. Nota para iterar VEREDICTOS.md.
- **El veredicto agregado 💨 HUMO NO es el promedio de los individuales.** Tomados uno a uno, solo 4 de 16 claims son problemáticos. Pero esos 4 son precisamente los que sostienen la tesis "la carrera no tiene freno" (GPT-6 inminente + super app + 40% mejora + ventana 2M). Si se quitan, la tesis del post pierde la mitad de su momentum. Por eso el conjunto es HUMO: no porque esté lleno de falsedades, sino porque el marco narrativo depende de los claims menos sostenidos.
- **Discrepancia detectada:** el post dice "14 de abril" pero la filtración original parece decir "16 de abril". Esta divergencia interna al propio corpus de rumores es una señal útil — cuando los rumores divergen entre sí, la confianza colectiva baja aún más.
- **Crédito al autor:** Pep Martorell atribuye explícitamente el "14 de abril" a "usuarios en Twitter". Ese es un nivel mínimo de honestidad epistémica que el sistema ahora reconoce con la sub-etiqueta 📎 **BIEN ATRIBUIDO**. Sin embargo, otros tres claims del mismo corpus de filtraciones (super app, 40%, 2M tokens, aviso privado al gobierno USA) se presentan sin atribución, lo que justifica 🎭 **RUMOR DISFRAZADO DE HECHO**. La honestidad epistémica global del autor es media, no baja.
