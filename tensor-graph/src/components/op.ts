

export function to_op_str(op: any): string {
    if (typeof op.op === 'string') {
        return op.op
    } else if ('Sum' === op.op) {
        let axes = op['Sum']['axes']
        return `Sum([${axes.join(',')}])`
    } else if ('Max' === op.op) {
        let axes = op['Max']['axes']
        return `Max([${axes.join(',')}])`
    } else if ('Min' === op.op) {
        let axes = op['Min']['axes']
        return `Min([${axes.join(',')}])`
    } else if ('Var' === op.op) {
        let axes = op['Var']['axes']
        return `Var([${axes.join(',')}])`
    } else if ('Mean' === op.op) {
        let axes = op['Mean']['axes']
        return `Mean([${axes.join(',')}])`
    } else if ('Transpose' === op.op) {
        let axes = op['Transpose']['axes']
        return `Transpose([${axes.join(',')}])`
    } else {
        return 'UnknownOp'
    }
}