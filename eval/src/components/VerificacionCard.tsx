'use client';

import { Card, Text, Group, Stack } from '@mantine/core';
import Link from 'next/link';
import type { VerificacionFrontmatter } from '../types/verificacion';
import { VerdictBadge } from './VerdictBadge';

interface VerificacionCardProps {
  slug: string;
  frontmatter: VerificacionFrontmatter;
}

export function VerificacionCard({ slug, frontmatter }: VerificacionCardProps) {
  return (
    <Card component={Link} href={`/verificaciones/${slug}`} withBorder shadow="sm" padding="lg" radius="md">
      <Stack gap="sm">
        <Group justify="space-between" align="flex-start">
          <VerdictBadge verdict={frontmatter.veredicto_agregado} size="md" />
          <Text size="sm" c="dimmed">
            {frontmatter.fecha_verificacion}
          </Text>
        </Group>
        <Text fw={600} size="lg" lineClamp={2}>
          {frontmatter.titulo}
        </Text>
        <Group gap="xs">
          <Text size="sm" c="dimmed">
            {frontmatter.plataforma}
          </Text>
          <Text size="sm" c="dimmed">
            ·
          </Text>
          <Text size="sm" c="dimmed">
            {frontmatter.autor_pieza}
          </Text>
        </Group>
      </Stack>
    </Card>
  );
}
