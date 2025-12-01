# GraphQL Generated Types

Este directorio contiene los tipos TypeScript generados automáticamente desde el schema GraphQL.

## Generación

Para generar los tipos, ejecuta:

```bash
npm run codegen
```

O en modo watch (regenera automáticamente cuando cambia el schema):

```bash
npm run codegen:watch
```

## Uso

En lugar de escribir queries GraphQL manualmente, usa los hooks generados:

```typescript
// ❌ Antes (Manual y propenso a error)
const { data } = useSWR('query { consciousness { phiValue } }')

// ✅ Después (Generado y seguro)
import { useGetConsciousnessStatusQuery } from './generated/graphql'
const { data } = useGetConsciousnessStatusQuery()
// data.consciousness.phiValue ahora tiene autocompletado y validación de tipos
```

## Beneficios

- ✅ Type safety: El compilador de TypeScript detecta errores antes del deploy
- ✅ Autocompletado: IDE sugiere campos disponibles
- ✅ Refactoring seguro: Cambios en el backend se reflejan automáticamente
- ✅ Documentación: Los tipos sirven como documentación viva

