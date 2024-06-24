

export function to_op_str(op: any): string {
    if (typeof op === 'string') {
        return op
    } else if ('Sum' in op) {
        let axes = op['Sum']['axes']
        return `Sum([${axes.join(',')}])`
    } else if ('Max' in op) {
        let axes = op['Max']['axes']
        return `Max([${axes.join(',')}])`
    } else if ('Min' in op) {
        let axes = op['Min']['axes']
        return `Min([${axes.join(',')}])`
    } else if ('Var' in op) {
        let axes = op['Var']['axes']
        return `Var([${axes.join(',')}])`
    } else if ('Mean' in op) {
        let axes = op['Mean']['axes']
        return `Mean([${axes.join(',')}])`
    } else if ('Transpose' in op) {
        let axes = op['Transpose']['axes']
        return `Transpose([${axes.join(',')}])`
    } else {
        return 'Unknown'
    }
}