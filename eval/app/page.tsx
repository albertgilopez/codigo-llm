import { Container, Title, Text, Stack, Group, Anchor, Divider } from '@mantine/core';
import { getAllVerificaciones } from '../src/lib/verificaciones';
import { VerificacionCard } from '../src/components/VerificacionCard';

export default function HomePage() {
  const verificaciones = getAllVerificaciones();

  return (
    <Container size="md" py="xl">
      <Stack gap="xl">
        <Stack gap="xs">
          <Group gap="xs" align="baseline">
            <Title order={1} fw={800}>
              eval
            </Title>
            <Text c="dimmed" size="sm">
              por{' '}
              <Anchor href="https://www.codigollm.es" target="_blank" size="sm">
                Codigo LLM
              </Anchor>
            </Text>
          </Group>
          <Text size="lg" fw={500}>
            Verificación de noticias sobre Inteligencia Artificial Generativa
          </Text>
          <Text c="dimmed">
            Esta sección de Codigo LLM está dedicada a la verificación de noticias, claims y rumores
            del ecosistema IA. Cada claim evaluado contra fuentes primarias para que sepas qué es
            real y qué es humo.
          </Text>
        </Stack>

        <Divider />

        {verificaciones.length === 0 ? (
          <Text c="dimmed">No hay verificaciones publicadas todavía.</Text>
        ) : (
          <Stack gap="md">
            <Text fw={600} size="lg">
              Últimas verificaciones
            </Text>
            {verificaciones.map((v) => (
              <VerificacionCard key={v.slug} slug={v.slug} frontmatter={v.frontmatter} />
            ))}
          </Stack>
        )}

        <Divider />

        <Group gap="lg">
          <Anchor href="/verificaciones" size="sm">
            Archivo completo
          </Anchor>
          <Anchor href="/metodologia" size="sm">
            Metodología
          </Anchor>
          <Anchor href="/sobre" size="sm">
            Sobre
          </Anchor>
          <Anchor href="https://www.codigollm.es" size="sm" target="_blank">
            codigollm.es
          </Anchor>
          <Anchor
            href="https://github.com/albertgilopez/codigo-llm/tree/main/eval"
            size="sm"
            target="_blank"
          >
            GitHub
          </Anchor>
        </Group>
      </Stack>
    </Container>
  );
}
