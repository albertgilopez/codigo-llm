import { Container, Title, Text, Stack, Divider, Badge, Group, Anchor } from '@mantine/core';
import { VERDICTS } from '../../src/lib/verdicts';
import type { VerdictKey } from '../../src/types/verificacion';

export const metadata = {
  title: 'Metodología',
  description:
    'Cómo verificamos noticias sobre IA: metodología SIFT, flujo Verificat, sistema de veredictos y jerarquía de fuentes T1-T4.',
  alternates: {
    canonical: 'https://eval.codigollm.es/metodologia',
    languages: { 'es-ES': 'https://eval.codigollm.es/metodologia' },
  },
};

export default function MetodologiaPage() {
  const verdictKeys = Object.keys(VERDICTS) as VerdictKey[];

  return (
    <Container size="md" py="xl">
      <Stack gap="xl">
        <Stack gap="xs">
          <Title order={1}>Metodología</Title>
          <Text c="dimmed">
            Cómo verificamos las noticias sobre Inteligencia Artificial Generativa.
          </Text>
        </Stack>

        <Stack gap="md">
          <Title order={2}>Base metodológica</Title>
          <Text>
            Este flujo adapta la metodología <strong>SIFT</strong> (Stop, Investigate, Find better
            coverage, Trace the original) y el <strong>flujo de verificación de Verificat</strong> (5
            pasos) al dominio específico de noticias sobre laboratorios de IA, modelos, benchmarks y
            el ecosistema tech.
          </Text>
          <Text>
            La metodología fue desarrollada durante la{' '}
            <Anchor
              href="https://www.uab.cat/web/postgrado/microcredencial-en-verificacion-de-la-informacion-y-fact-checking-herramientas-para-el-periodismo-en-el-ecosistema-digital/informacion-general-1206597475768.html/param1-5173_es/"
              target="_blank"
            >
              Microcredencial en Verificación de la Información y Fact-Checking
            </Anchor>{' '}
            de la UAB en colaboración con Verificat.
          </Text>
        </Stack>

        <Divider />

        <Stack gap="md">
          <Title order={2}>Sistema de veredictos</Title>
          <Text>
            Tratamos cada post o noticia como si fuera la salida de un LLM y le hacemos un{' '}
            <em>eval</em>. La simetría{' '}
            <strong>SIN ALUCINACIONES ↔ CON ALUCINACIONES</strong> es deliberada.
          </Text>

          <Stack gap="sm">
            {verdictKeys.map((key) => {
              const v = VERDICTS[key];
              return (
                <Group key={key} gap="md" align="center">
                  <Badge color={v.color} variant="filled" size="lg" w={240}>
                    {v.emoji} {v.label}
                  </Badge>
                  <Text size="sm" fs="italic" style={{ flex: 1 }}>
                    {v.description}
                  </Text>
                </Group>
              );
            })}
          </Stack>
        </Stack>

        <Divider />

        <Stack gap="md">
          <Title order={2}>Jerarquía de fuentes</Title>
          <Text>
            No todas las fuentes pesan igual. Para el dominio IA usamos una jerarquía de 4 niveles:
          </Text>

          <Stack gap="sm">
            <Group gap="md" align="flex-start">
              <Badge color="green" variant="light" size="md" w={60}>
                T1
              </Badge>
              <Stack gap={2} style={{ flex: 1 }}>
                <Text fw={600} size="sm">Primarias oficiales</Text>
                <Text size="sm" c="dimmed">
                  System cards, blog posts oficiales del lab, release notes, papers arXiv del lab
                </Text>
              </Stack>
            </Group>
            <Group gap="md" align="flex-start">
              <Badge color="blue" variant="light" size="md" w={60}>
                T2
              </Badge>
              <Stack gap={2} style={{ flex: 1 }}>
                <Text fw={600} size="sm">Secundarias tier 1</Text>
                <Text size="sm" c="dimmed">
                  TechCrunch, CNBC, Fortune, NYT, The Hacker News, MIT Tech Review
                </Text>
              </Stack>
            </Group>
            <Group gap="md" align="flex-start">
              <Badge color="yellow" variant="light" size="md" w={60}>
                T3
              </Badge>
              <Stack gap={2} style={{ flex: 1 }}>
                <Text fw={600} size="sm">Secundarias especializadas</Text>
                <Text size="sm" c="dimmed">
                  Transformer News, Import AI, LessWrong, Ars Technica, The Verge
                </Text>
              </Stack>
            </Group>
            <Group gap="md" align="flex-start">
              <Badge color="red" variant="light" size="md" w={60}>
                T4
              </Badge>
              <Stack gap={2} style={{ flex: 1 }}>
                <Text fw={600} size="sm">Terciarias (requieren corroboración)</Text>
                <Text size="sm" c="dimmed">
                  Twitter/X, Polymarket, Reddit, Medium, Substack individuales, leaks
                </Text>
              </Stack>
            </Group>
          </Stack>

          <Text size="sm" c="dimmed" mt="sm">
            Regla de oro: un claim solo puede ser ✅ SIN ALUCINACIONES con al menos 1 fuente T1 o 2
            fuentes T2 independientes. Claims sustentados solo por T3-T4 son automáticamente 🐦
            RUMOR DE X.
          </Text>
        </Stack>

        <Divider />

        <Stack gap="md">
          <Title order={2}>Qué NO es este sitio</Title>
          <Stack gap="xs">
            <Text>
              <strong>No es un desmentido al autor.</strong> Verificamos las noticias e informaciones,
              no a las personas.
            </Text>
            <Text>
              <strong>No es periodismo de opinión.</strong> Las opiniones se etiquetan como tales, no
              se verifican como hechos.
            </Text>
            <Text>
              <strong>No es automático.</strong> El juicio humano decide veredictos. Las herramientas
              solo reúnen evidencia.
            </Text>
          </Stack>
        </Stack>

        <Group gap="lg" mt="xl">
          <Anchor href="/" size="sm">
            ← Inicio
          </Anchor>
          <Anchor href="/sobre" size="sm">
            Sobre este proyecto
          </Anchor>
        </Group>
      </Stack>
    </Container>
  );
}
