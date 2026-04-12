import { Container, Title, Text, Stack } from '@mantine/core';
import { getAllVerificaciones } from '../../src/lib/verificaciones';
import { VerificacionCard } from '../../src/components/VerificacionCard';

export const metadata = {
  title: 'Verificaciones',
  description: 'Archivo completo de verificaciones de noticias sobre Inteligencia Artificial Generativa. Cada claim evaluado contra fuentes primarias.',
  alternates: {
    canonical: 'https://eval.codigollm.es/verificaciones',
    languages: { 'es-ES': 'https://eval.codigollm.es/verificaciones' },
  },
};

export default function VerificacionesPage() {
  const verificaciones = getAllVerificaciones();

  return (
    <Container size="md" py="xl">
      <Stack gap="xl">
        <Stack gap="xs">
          <Title order={1}>Verificaciones</Title>
          <Text c="dimmed">
            {verificaciones.length} verificacion{verificaciones.length !== 1 ? 'es' : ''}{' '}
            publicada{verificaciones.length !== 1 ? 's' : ''}
          </Text>
        </Stack>

        <Stack gap="md">
          {verificaciones.map((v) => (
            <VerificacionCard key={v.slug} slug={v.slug} frontmatter={v.frontmatter} />
          ))}
        </Stack>
      </Stack>
    </Container>
  );
}
