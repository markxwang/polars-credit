use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn woe_output(_: &[Field]) -> PolarsResult<Field> {
    let value = Field::new("value", DataType::String);
    let woe: Field = Field::new("woe", DataType::Float64);
    let v: Vec<Field> = vec![value, woe];
    Ok(Field::new("woe_output", DataType::Struct(v)))
}

fn cal_woe(x: &Series, y: &Series) -> PolarsResult<LazyFrame> {
    let df = df!(
        "x" => x,
        "y" => y,
    )?;

    let out = df
        .lazy()
        .group_by([col("x")])
        .agg([
            col("y").eq(lit(1)).sum().alias("bad"),
            col("y").eq(lit(0)).sum().alias("good"),
        ])
        .select([
            col("x"),
            ((col("good")).cast(DataType::Float64) / (col("good").sum()).cast(DataType::Float64)),
            ((col("bad")).cast(DataType::Float64) / (col("bad").sum()).cast(DataType::Float64)),
        ])
        .with_column(
            (col("bad") / col("good"))
                .log(std::f64::consts::E)
                .alias("woe"),
        );
    Ok(out)
}

#[polars_expr(output_type_func=woe_output)]
fn pl_woe_discrete(inputs: &[Series]) -> PolarsResult<Series> {
    let df = cal_woe(&inputs[0], &inputs[1])?
        .select([col("x"), col("woe")])
        .collect()?;

    Ok(df.into_struct("woe_output").into_series())
}

#[polars_expr(output_type=Float64)]
fn pl_iv(inputs: &[Series]) -> PolarsResult<Series> {
    let df_iv = cal_woe(&inputs[0], &inputs[1])?
        .select([((col("bad") - col("good")) * col("woe")).sum().alias("iv")])
        .collect()?;
    let iv_series = df_iv.column("iv")?.clone();
    Ok(iv_series)
}
