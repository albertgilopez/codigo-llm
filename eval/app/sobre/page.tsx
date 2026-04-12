import { Container, Title, Text, Stack, Divider, Anchor, Group } from '@mantine/core';

export const metadata = {
  title: 'Sobre',
  description: 'Quién mantiene eval.codigollm.es: Albert Gil López, CTO de M.IA, profesor UAB, graduado en fact-checking UAB-Verificat.',
  alternates: {
    canonical: 'https://eval.codigollm.es/sobre',
    languages: { 'es-ES': 'https://eval.codigollm.es/sobre' },
  },
};

export default function SobrePage() {
  return (
    <Container size="md" py="xl">
      <Stack gap="xl">
        <Stack gap="xs">
          <Title order={1}>Sobre este proyecto</Title>
          <Text c="dimmed">Quién lo mantiene, por qué existe, y cómo contactar.</Text>
        </Stack>

        <Stack gap="md">
          <Title order={2}>Qué es eval</Title>
          <Text>
            <strong>eval</strong> es la sección de verificación de noticias de{' '}
            <Anchor href="https://www.codigollm.es" target="_blank">
              Codigo LLM
            </Anchor>

            . Esta sección está dedicada a la verificación de noticias, claims y rumores del
            ecosistema de Inteligencia Artificial Generativa para que sepas qué es real y qué es
            humo.
          </Text>
          <Text>
            Cada verificación evalúa los claims de una pieza (post, artículo, hilo) contra fuentes
            primarias oficiales, aplicando una metodología adaptada de SIFT y del flujo de
            verificación de Verificat.
          </Text>
        </Stack>

        <Divider />

        <Stack gap="md">
          <Title order={2}>Quién lo mantiene</Title>
          <Text>
            <strong>Albert Gil López</strong> — Ingeniero informático (UAB, 2018), CTO de{' '}
            <Anchor href="https://predicta.es" target="_blank">
              M.IA
            </Anchor>{' '}
            y profesor asociado en la Escola d&apos;Enginyeria UAB.
          </Text>
          <Text>
            Actualmente cursando el Máster en Filosofía para los Retos Contemporáneos (UOC) y
            graduado de la Microcredencial en Verificación de la Información y Fact-Checking
            (UAB-Verificat, 2026).
          </Text>
        </Stack>

        <Divider />

        <Stack gap="md">
          <Title order={2}>Derecho de respuesta</Title>
          <Text>
            Si eres autor de una pieza que hemos verificado y quieres aportar contexto, corregir
            datos o ejercer tu derecho de respuesta, contacta en{' '}
            <Anchor href="mailto:soporte@codigollm.es">soporte@codigollm.es</Anchor>. Todas las
            respuestas relevantes se incorporarán a la verificación correspondiente.
          </Text>
        </Stack>

        <Divider />

        <Stack gap="md">
          <Title order={2}>Contacto</Title>
          <Group gap="lg">
            <Anchor href="https://www.codigollm.es" target="_blank" size="sm">
              codigollm.es
            </Anchor>
            <Anchor href="https://albertgilopez.com" target="_blank" size="sm">
              albertgilopez.com
            </Anchor>
            <Anchor href="https://linkedin.com/in/albertgilopez" target="_blank" size="sm">
              LinkedIn
            </Anchor>
            <Anchor href="https://github.com/albertgilopez" target="_blank" size="sm">
              GitHub
            </Anchor>
            <Anchor href="mailto:soporte@codigollm.es" size="sm">
              soporte@codigollm.es
            </Anchor>
          </Group>
        </Stack>

        <Group gap="lg" mt="xl">
          <Anchor href="/" size="sm">
            ← Inicio
          </Anchor>
          <Anchor href="/metodologia" size="sm">
            Metodología
          </Anchor>
        </Group>
      </Stack>
    </Container>
  );
}
