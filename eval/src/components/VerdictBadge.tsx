'use client';

import { Badge, Tooltip } from '@mantine/core';
import type { VerdictKey } from '../types/verificacion';
import { VERDICTS } from '../lib/verdicts';

interface VerdictBadgeProps {
  verdict: VerdictKey;
  size?: 'sm' | 'md' | 'lg' | 'xl';
}

export function VerdictBadge({ verdict, size = 'lg' }: VerdictBadgeProps) {
  const info = VERDICTS[verdict];
  if (!info) return null;

  return (
    <Tooltip label={info.description} withArrow>
      <Badge color={info.color} size={size} variant="filled" radius="sm">
        {info.emoji} {info.label}
      </Badge>
    </Tooltip>
  );
}
