export enum Classification {
    BENIGN = 'BENIGN',
    MALIGNANT = 'MALIGNANT',
    MALIGNANT_WITH_CALLBACK = 'MALIGNANT_WITH_CALLBACK',
};

export type Result = {
    classification?: keyof typeof Classification;
};
