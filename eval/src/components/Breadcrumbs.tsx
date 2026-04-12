'use client';

import { Breadcrumbs as MantineBreadcrumbs, Anchor, Text } from '@mantine/core';

interface BreadcrumbsProps {
  items: { label: string; href?: string }[];
}

export function Breadcrumbs({ items }: BreadcrumbsProps) {
  return (
    <MantineBreadcrumbs mb="md">
      {items.map((item, i) =>
        item.href ? (
          <Anchor href={item.href} size="sm" key={i}>
            {item.label}
          </Anchor>
        ) : (
          <Text size="sm" c="dimmed" key={i}>
            {item.label}
          </Text>
        )
      )}
    </MantineBreadcrumbs>
  );
}
